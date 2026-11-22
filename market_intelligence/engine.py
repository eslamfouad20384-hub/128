"""Statistical EGX market-regime and forward-outcome engine.

This module is deliberately advisory-only. It reads already-loaded daily OHLCV
frames and never changes the host application's entries, exits, stops, targets,
position sizing, or portfolio records.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ENGINE_VERSION = "EMPI-1.0"
WEIGHTS = {
    "breadth": 25.0,
    "liquidity": 20.0,
    "trend": 20.0,
    "momentum": 15.0,
    "risk": 10.0,
}
# Sector contribution is intentionally omitted until a verified point-in-time
# sector map exists. Available weights are normalized to 100.


def _safe_float(value, default=np.nan):
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _market_label(score: float) -> str:
    if not np.isfinite(score):
        return "بيانات غير كافية"
    if score < 20:
        return "هبوط حاد"
    if score < 40:
        return "ضغط بيعي"
    if score < 60:
        return "عرضي / محايد"
    if score < 75:
        return "صعود محتمل"
    if score < 90:
        return "صعود قوي"
    return "صعود استثنائي"


def _pressure_label(row: pd.Series) -> str:
    breadth = _safe_float(row.get("breadth_50"))
    pressure = _safe_float(row.get("pressure_score"))
    ret20 = _safe_float(row.get("median_return_20"))
    if breadth >= 60 and pressure >= 60 and ret20 > 0:
        return "تأكيد صعود"
    if breadth >= 50 and pressure >= 55 and ret20 <= 0:
        return "تجميع محتمل"
    if breadth < 35 and pressure < 40:
        return "ضغط هبوط"
    if breadth >= 65 and pressure < 45:
        return "ضعف / تباعد صاعد"
    if breadth >= 65 and ret20 > 0.08:
        return "صعود ناضج"
    return "مرحلة انتقالية"


def _make_symbol_panel(
    daily: pd.DataFrame,
    symbols: Iterable[str],
    extract_symbol_data: Callable,
    add_indicators: Callable,
    exclude_daily_bar: Callable,
) -> tuple[pd.DataFrame, dict]:
    pieces = []
    stats = {"requested": 0, "usable": 0, "excluded": 0, "errors": 0}
    for symbol in symbols:
        stats["requested"] += 1
        try:
            frame = extract_symbol_data(daily, symbol)
            frame = exclude_daily_bar(frame)
            if frame is None or frame.empty or len(frame) < 60:
                stats["excluded"] += 1
                continue
            frame = add_indicators(frame.copy())
            if frame is None or frame.empty:
                stats["excluded"] += 1
                continue
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(frame.columns):
                stats["excluded"] += 1
                continue
            x = pd.DataFrame(index=frame.index)
            close = pd.to_numeric(frame["Close"], errors="coerce")
            high = pd.to_numeric(frame["High"], errors="coerce")
            low = pd.to_numeric(frame["Low"], errors="coerce")
            volume = pd.to_numeric(frame["Volume"], errors="coerce").clip(lower=0)
            x["symbol"] = str(symbol).replace(".CA", "")
            x["close"] = close
            x["ret1"] = close.pct_change(1)
            x["ret5"] = close.pct_change(5)
            x["ret10"] = close.pct_change(10)
            x["ret20"] = close.pct_change(20)
            x["ret60"] = close.pct_change(60)
            for source, target in [("ema20", "above20"), ("ema50", "above50"), ("ema200", "above200")]:
                x[target] = (close > pd.to_numeric(frame.get(source), errors="coerce")).where(
                    pd.to_numeric(frame.get(source), errors="coerce").notna()
                )
            x["rsi"] = pd.to_numeric(frame.get("rsi"), errors="coerce")
            x["adx"] = pd.to_numeric(frame.get("adx"), errors="coerce")
            x["atr_pct"] = pd.to_numeric(frame.get("atr_pct"), errors="coerce")
            x["mfi"] = pd.to_numeric(frame.get("mfi"), errors="coerce")
            x["obv_slope"] = pd.to_numeric(frame.get("obv_slope"), errors="coerce")
            bar_range = (high - low).replace(0, np.nan)
            x["close_location"] = ((2 * close - high - low) / bar_range).clip(-1, 1)
            x["turnover_proxy"] = close * volume
            x["turnover_ma20"] = x["turnover_proxy"].rolling(20, min_periods=10).median()
            x["liquidity_ratio"] = x["turnover_proxy"] / x["turnover_ma20"].replace(0, np.nan)
            x["new_high20"] = close.ge(close.rolling(20, min_periods=15).max())
            x["new_low20"] = close.le(close.rolling(20, min_periods=15).min())
            x = x.replace([np.inf, -np.inf], np.nan)
            x = x.dropna(subset=["close", "ret1"])
            if not x.empty:
                pieces.append(x)
                stats["usable"] += 1
            else:
                stats["excluded"] += 1
        except Exception:
            stats["errors"] += 1
    if not pieces:
        return pd.DataFrame(), stats
    panel = pd.concat(pieces, axis=0).reset_index(names="date")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.tz_localize(None)
    panel = panel.dropna(subset=["date"]).sort_values(["date", "symbol"])
    return panel, stats


def _aggregate_market(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for date, g in panel.groupby("date", sort=True):
        valid = g["ret1"].notna()
        ret = g.loc[valid, "ret1"]
        turnover = g["turnover_proxy"].where(g["turnover_proxy"] >= 0, np.nan)
        turnover_total = turnover.sum(min_count=1)
        advancing_turnover = turnover[g["ret1"] > 0].sum(min_count=1)
        above20 = g["above20"].dropna()
        above50 = g["above50"].dropna()
        above200 = g["above200"].dropna()
        close_loc = g["close_location"].dropna()
        pressure_parts = []
        if not close_loc.empty:
            pressure_parts.append((close_loc.mean() + 1) * 50)
        mfi = g["mfi"].dropna()
        if not mfi.empty:
            pressure_parts.append(float(np.clip(mfi.mean(), 0, 100)))
        obv = g["obv_slope"].dropna()
        if not obv.empty:
            pressure_parts.append(float((obv.gt(0).mean()) * 100))
        pressure_score = float(np.mean(pressure_parts)) if pressure_parts else np.nan
        liquidity_ratio = g["liquidity_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        atr = g["atr_pct"].dropna()
        adv = int((ret > 0).sum())
        dec = int((ret < 0).sum())
        unchanged = int((ret == 0).sum())
        valid_n = int(valid.sum())
        rows.append({
            "date": date,
            "symbols_valid": valid_n,
            "advancers": adv,
            "decliners": dec,
            "unchanged": unchanged,
            "advance_decline_ratio": adv / max(dec, 1),
            "breadth_20": float(above20.mean() * 100) if not above20.empty else np.nan,
            "breadth_50": float(above50.mean() * 100) if not above50.empty else np.nan,
            "breadth_200": float(above200.mean() * 100) if not above200.empty else np.nan,
            "new_high20_pct": float(g["new_high20"].dropna().mean() * 100) if g["new_high20"].notna().any() else np.nan,
            "new_low20_pct": float(g["new_low20"].dropna().mean() * 100) if g["new_low20"].notna().any() else np.nan,
            "median_return_1": float(ret.median()) if not ret.empty else np.nan,
            "median_return_5": float(g["ret5"].median()) if g["ret5"].notna().any() else np.nan,
            "median_return_10": float(g["ret10"].median()) if g["ret10"].notna().any() else np.nan,
            "median_return_20": float(g["ret20"].median()) if g["ret20"].notna().any() else np.nan,
            "median_return_60": float(g["ret60"].median()) if g["ret60"].notna().any() else np.nan,
            "turnover_proxy": float(turnover_total) if np.isfinite(turnover_total) else np.nan,
            "advancing_turnover_share": float(advancing_turnover / turnover_total * 100) if np.isfinite(turnover_total) and turnover_total > 0 and np.isfinite(advancing_turnover) else np.nan,
            "median_liquidity_ratio": float(liquidity_ratio.median()) if not liquidity_ratio.empty else np.nan,
            "top5_turnover_concentration": float(turnover.nlargest(5).sum() / turnover_total * 100) if np.isfinite(turnover_total) and turnover_total > 0 else np.nan,
            "pressure_score": pressure_score,
            "median_atr_pct": float(atr.median()) if not atr.empty else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # Components are scored on transparent, bounded scales. Missing inputs remain missing.
    out["breadth_score"] = (
        out[["breadth_20", "breadth_50", "breadth_200"]].mean(axis=1, skipna=True)
    )
    liq = 50 + (out["median_liquidity_ratio"] - 1.0).clip(-1.0, 2.0) * 15
    if out["advancing_turnover_share"].notna().any():
        liq = liq + ((out["advancing_turnover_share"] - 50).clip(-50, 50) * 0.25)
    # Missing liquidity is excluded from the EMPI denominator, never replaced by neutral.
    out["liquidity_score"] = liq.where(out["median_liquidity_ratio"].notna()).clip(0, 100)
    trend_raw = 50 + out["median_return_20"].clip(-0.20, 0.20) * 125 + out["median_return_60"].clip(-0.40, 0.40) * 25
    out["trend_score"] = trend_raw.clip(0, 100)
    momentum_raw = 50 + out["median_return_5"].clip(-0.10, 0.10) * 150 + out["median_return_10"].clip(-0.15, 0.15) * 75
    if out["pressure_score"].notna().any():
        momentum_raw = momentum_raw * 0.70 + out["pressure_score"].fillna(50) * 0.30
    out["momentum_score"] = momentum_raw.clip(0, 100)
    out["risk_score"] = (100 - out["median_atr_pct"].clip(lower=0, upper=0.15) * (100 / 0.15)).clip(0, 100)
    score_cols = {"breadth": "breadth_score", "liquidity": "liquidity_score", "trend": "trend_score", "momentum": "momentum_score", "risk": "risk_score"}
    weighted_sum = pd.Series(0.0, index=out.index)
    weight_sum = pd.Series(0.0, index=out.index)
    for key, col in score_cols.items():
        valid = out[col].notna()
        weighted_sum.loc[valid] += out.loc[valid, col] * WEIGHTS[key]
        weight_sum.loc[valid] += WEIGHTS[key]
    out["empi"] = (weighted_sum / weight_sum.replace(0, np.nan)).clip(0, 100)
    out["market_regime"] = out["empi"].apply(_market_label)
    out["pressure_phase"] = out.apply(_pressure_label, axis=1)
    out["market_proxy_return"] = out["median_return_1"]
    return out


def _forward_returns(market: pd.DataFrame, horizon: int) -> pd.Series:
    r = pd.to_numeric(market["market_proxy_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(r)
    out = np.full(n, np.nan, dtype=float)
    # Return from the next session through the requested horizon (no same-day leakage).
    log_r = np.log1p(np.clip(r, -0.95, None))
    cumulative = np.concatenate(([0.0], np.cumsum(log_r)))
    for i in range(max(0, n - horizon)):
        start = i + 1
        end = i + horizon + 1
        if end <= n:
            out[i] = np.expm1(cumulative[end] - cumulative[start])
    return pd.Series(out, index=market.index)


def _predict_horizon(market: pd.DataFrame, horizon: int) -> dict:
    if len(market) < horizon + 25:
        return {"horizon": horizon, "status": "تاريخ غير كافٍ", "samples": 0}
    matured = market.iloc[: max(0, len(market) - horizon)].copy()
    outcomes = _forward_returns(market, horizon).reindex(matured.index)
    matured["future_return"] = outcomes
    matured = matured.dropna(subset=["future_return", "empi", "breadth_score"])
    current = market.iloc[-1]
    if matured.empty:
        return {"horizon": horizon, "status": "لا توجد عينات مكتملة", "samples": 0}
    # Analog regime: similar EMPI and breadth. Widen once, never fabricate samples.
    analogs = matured[
        (matured["empi"].sub(float(current["empi"])).abs() <= 10)
        & (matured["breadth_score"].sub(float(current["breadth_score"])).abs() <= 18)
    ]
    if len(analogs) < 20:
        analogs = matured[matured["empi"].sub(float(current["empi"])).abs() <= 15]
    if len(analogs) < 20:
        return {"horizon": horizon, "status": "عينات تاريخية مشابهة غير كافية", "samples": int(len(analogs))}
    values = analogs["future_return"].astype(float)
    up = float((values > 0.01).mean() * 100)
    down = float((values < -0.01).mean() * 100)
    sideways = max(0.0, 100.0 - up - down)
    return {
        "horizon": horizon,
        "status": "تقدير إحصائي تاريخي - غير مضمون",
        "samples": int(len(values)),
        "up_probability": round(up, 1),
        "down_probability": round(down, 1),
        "sideways_probability": round(sideways, 1),
        "mean_return": round(float(values.mean() * 100), 2),
        "median_return": round(float(values.median() * 100), 2),
        "best_return": round(float(values.max() * 100), 2),
        "worst_return": round(float(values.min() * 100), 2),
    }


def _store_snapshot(row: pd.Series, coverage: dict, db_path: Path) -> None:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: (None if pd.isna(v) else float(v) if isinstance(v, (np.floating, float, int, np.integer)) else str(v)) for k, v in row.to_dict().items()}
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS market_snapshots (
                snapshot_date TEXT PRIMARY KEY, engine_version TEXT NOT NULL,
                empi REAL, regime TEXT, breadth_20 REAL, breadth_50 REAL,
                breadth_200 REAL, turnover_proxy REAL, coverage_json TEXT,
                snapshot_json TEXT NOT NULL)""")
            conn.execute("""INSERT OR REPLACE INTO market_snapshots
                (snapshot_date, engine_version, empi, regime, breadth_20, breadth_50,
                 breadth_200, turnover_proxy, coverage_json, snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(pd.Timestamp(row["date"]).date()), ENGINE_VERSION,
                 _safe_float(row.get("empi")), str(row.get("market_regime", "")),
                 _safe_float(row.get("breadth_20")), _safe_float(row.get("breadth_50")),
                 _safe_float(row.get("breadth_200")), _safe_float(row.get("turnover_proxy")),
                 json.dumps(coverage, ensure_ascii=False), json.dumps(payload, ensure_ascii=False)))
            conn.commit()
    except Exception:
        # Storage is supplementary; never stop the stock scanner because of it.
        return


def build_market_intelligence(
    daily: pd.DataFrame,
    symbols: Iterable[str],
    extract_symbol_data: Callable,
    add_indicators: Callable,
    exclude_daily_bar: Callable,
    db_path: str | Path | None = None,
) -> dict:
    panel, coverage = _make_symbol_panel(daily, symbols, extract_symbol_data, add_indicators, exclude_daily_bar)
    market = _aggregate_market(panel)
    if market.empty:
        return {"ok": False, "message": "لا توجد بيانات يومية صالحة لبناء قراءة للسوق.", "coverage": coverage}
    latest = market.iloc[-1]
    predictions = {h: _predict_horizon(market, h) for h in (5, 10, 20)}
    if db_path is not None:
        _store_snapshot(latest, coverage, Path(db_path))
    history = market.tail(90).copy()
    return {
        "ok": True,
        "engine_version": ENGINE_VERSION,
        "latest": latest.to_dict(),
        "predictions": predictions,
        "coverage": coverage,
        "history": history,
        "history_rows": int(len(market)),
        "sector_data_available": False,
        "advisory": _advisory(latest),
    }


def _advisory(row: pd.Series) -> dict:
    score = _safe_float(row.get("empi"))
    if not np.isfinite(score):
        return {"exposure": "غير محدد", "note": "البيانات غير كافية؛ لا تستخدم المحرك لتغيير المخاطرة."}
    if score < 35:
        return {"exposure": "حذر مرتفع", "note": "راجع جودة الفرص والسيولة، واحتفظ بمرونة نقدية. لا يتم تعديل حجم الصفقة تلقائيًا."}
    if score < 50:
        return {"exposure": "حذر", "note": "انتقِ الإشارات الأعلى جودة وتجنب التوسع في المراكز الضعيفة."}
    if score < 65:
        return {"exposure": "متوازن", "note": "السوق محايد إلى إيجابي؛ الالتزام بقواعد المحرك الأساسي كما هي."}
    return {"exposure": "إيجابي بحذر", "note": "اتساع السوق داعم نسبيًا؛ لا يعني ذلك رفع مخاطرة الصفقة أو تجاوز حد المركز."}


def render_market_dashboard(st, result: dict) -> None:
    st.subheader("🧭 EGX Market Intelligence | ذكاء حالة السوق")
    st.caption("محرك إحصائي استرشادي مستقل. لا يغير إشارات الدخول أو الخروج أو الوقف أو الأهداف أو حجم المركز أو سجل المحفظة.")
    if not result or not result.get("ok"):
        st.warning((result or {}).get("message", "لا تتوفر قراءة للسوق."))
        return
    row = result["latest"]
    score = _safe_float(row.get("empi"))
    cols = st.columns(4)
    cols[0].metric("EMPI | ضغط السوق", f"{score:.1f}/100" if np.isfinite(score) else "—")
    cols[1].metric("حالة السوق", str(row.get("market_regime", "—")))
    cols[2].metric("اتساع فوق EMA50", f"{_safe_float(row.get('breadth_50')):.1f}%" if np.isfinite(_safe_float(row.get("breadth_50"))) else "—")
    cols[3].metric("الضغط الشرائي/البيعي", str(row.get("pressure_phase", "—")))

    st.markdown("#### مكونات قراءة السوق")
    comp_cols = st.columns(5)
    metrics = [
        ("Breadth", "breadth_score"), ("Liquidity", "liquidity_score"),
        ("Trend", "trend_score"), ("Momentum", "momentum_score"), ("Risk", "risk_score")
    ]
    for col, (label, key) in zip(comp_cols, metrics):
        value = _safe_float(row.get(key))
        col.metric(label, f"{value:.1f}/100" if np.isfinite(value) else "—")

    st.markdown("#### اتساع السوق والسيولة")
    breadth_cols = st.columns(4)
    for col, label, key in [
        (breadth_cols[0], "أسهم صاعدة", "advancers"),
        (breadth_cols[1], "أسهم هابطة", "decliners"),
        (breadth_cols[2], "فوق EMA20", "breadth_20"),
        (breadth_cols[3], "فوق EMA200", "breadth_200"),
    ]:
        value = _safe_float(row.get(key))
        suffix = "%" if key.startswith("breadth") else ""
        col.metric(label, f"{value:.1f}{suffix}" if np.isfinite(value) else "—")
    st.caption("قيمة التداول المعروضة هي Turnover Proxy = الإغلاق × الحجم، وليست قيمة تداول رسمية من البورصة.")
    turnover = _safe_float(row.get("turnover_proxy"))
    adv_share = _safe_float(row.get("advancing_turnover_share"))
    liq_cols = st.columns(3)
    liq_cols[0].metric("Turnover Proxy", f"{turnover:,.0f}" if np.isfinite(turnover) else "—")
    liq_cols[1].metric("حصة سيولة الأسهم الصاعدة", f"{adv_share:.1f}%" if np.isfinite(adv_share) else "—")
    concentration = _safe_float(row.get("top5_turnover_concentration"))
    liq_cols[2].metric("تركيز أكبر 5 أسهم", f"{concentration:.1f}%" if np.isfinite(concentration) else "—")

    st.markdown("#### التوقعات الإحصائية التاريخية")
    prediction_rows = []
    for horizon in (5, 10, 20):
        p = result["predictions"].get(horizon, {})
        prediction_rows.append({
            "الأفق": f"{horizon} جلسات", "الحالة": p.get("status", "—"),
            "عدد الحالات المشابهة": p.get("samples", 0),
            "احتمال الصعود %": p.get("up_probability", np.nan),
            "احتمال الهبوط %": p.get("down_probability", np.nan),
            "احتمال العرضي %": p.get("sideways_probability", np.nan),
            "متوسط العائد التاريخي %": p.get("mean_return", np.nan),
            "وسيط العائد التاريخي %": p.get("median_return", np.nan),
        })
    st.dataframe(pd.DataFrame(prediction_rows), use_container_width=True, hide_index=True)
    st.caption("التوقعات مبنية على تشابه EMPI واتساع السوق، وتُعرض فقط عند توافر 20 حالة تاريخية مكتملة على الأقل. العوائد مستقبلية داخل العينة التاريخية وليست ضمانًا.")

    st.markdown("#### التوصية الاسترشادية الموازية")
    advisory = result.get("advisory", {})
    st.info(f"**مستوى التعرض المقترح:** {advisory.get('exposure', '—')}\n\n{advisory.get('note', '')}")

    st.markdown("#### جودة البيانات والتغطية")
    cov = result.get("coverage", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("رموز القائمة الحالية", int(cov.get("requested", 0)))
    c2.metric("الرموز الصالحة", int(cov.get("usable", 0)))
    c3.metric("الفجوة عن المستهدف", int(cov.get("universe_gap", 0)))
    c4.metric("المستبعدة", int(cov.get("excluded", 0)))
    st.caption(f"المستهدف المسجل في الكود: {cov.get('target_universe_size', '—')} رمزًا. لا يتم اختلاق رموز لسد فجوة التغطية.")
    st.caption(f"أخطاء قراءة البيانات: {int(cov.get('errors', 0))}")
    st.caption(f"تاريخ البيانات: {pd.Timestamp(row['date']).date()} | تاريخ السوق المتاح: {result.get('history_rows', 0)} جلسة | القطاعات: غير مفعلة لعدم وجود خريطة قطاعية موثقة في هذه النسخة.")

    hist = result.get("history")
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        st.markdown("#### تطور مؤشر EMPI")
        chart = hist.set_index("date")[["empi"]].rename(columns={"empi": "EMPI"})
        st.line_chart(chart, use_container_width=True)
