# -*- coding: utf-8 -*-
"""估值指标 calc_valuation 移植 —— 与 skill 版逐字一致。

来源:ad_fundamental_analysis/scripts/run_fundamental_analysis.py(skill 版,依赖 AmazingData)。
本模块:纯 pandas/numpy,去掉 AmazingData;数据由 MCP(星耀数智)喂入。
对外入口:
  compute_valuation(code, begin_date, end_date) -> pd.DataFrame  12 列日频估值时序
  latest_valuation(code, begin_date, end_date) -> list[dict]     最新一期 12 字段
"""
import asyncio
import re

import numpy as np
import pandas as pd

from tradingagents.dataflows.providers.china import mcp_client

# ============================================================
# 基础 helper(与 skill 逐字一致,来源见文件头)
# ============================================================


def safe_div(a, b):
    """安全除法，分母为0或NaN时返回NaN"""
    a = pd.Series(a).values.astype(float) if not isinstance(a, np.ndarray) else a.astype(float)
    b = pd.Series(b).values.astype(float) if not isinstance(b, np.ndarray) else b.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((b == 0) | np.isnan(b) | np.isnan(a), np.nan, a / b)
    return result


def get_ttm(df, field):
    """计算TTM（滚动12个月累计值）
    Q1报告期: TTM = Q1本期 + 去年年报 - 去年Q1
    Q2报告期: TTM = Q2本期 + 去年年报 - 去年Q2
    Q3报告期: TTM = Q3本期 + 去年年报 - 去年Q3
    Q4报告期(年报): TTM = 年报值
    """
    if df is None or df.empty or field not in df.columns:
        return pd.Series(dtype=float)
    df = df.sort_values('REPORTING_PERIOD').reset_index(drop=True)
    rp = df['REPORTING_PERIOD'].astype(str)
    result = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        val = df[field].iloc[i]
        if pd.isna(val):
            continue
        rp_str = rp.iloc[i]
        yr = rp_str[:4]
        mmdd = rp_str[4:]
        if mmdd == '1231':
            result.iloc[i] = val
        else:
            prev_yr = str(int(yr) - 1)
            ann_mask = rp == prev_yr + '1231'
            same_mask = rp == prev_yr + mmdd
            if ann_mask.any() and same_mask.any():
                ann_val = df.loc[ann_mask, field].iloc[-1]
                same_val = df.loc[same_mask, field].iloc[-1]
                if pd.notna(ann_val) and pd.notna(same_val):
                    result.iloc[i] = val + ann_val - same_val
    return result


def _filter_statements(df):
    """过滤财务报表：只保留合并报表（STATEMENT_TYPE='1'），同一报告期取最新记录。"""
    if df is None or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if 'STATEMENT_TYPE' in df.columns:
        st = df['STATEMENT_TYPE'].astype(str)
        mask &= st == '1'
    filtered = df[mask].copy()
    if filtered.empty:
        return df
    if 'ACTUAL_ANN_DATE' in filtered.columns and 'REPORTING_PERIOD' in filtered.columns:
        filtered = filtered.sort_values(['REPORTING_PERIOD', 'ACTUAL_ANN_DATE'])
        filtered = filtered.drop_duplicates('REPORTING_PERIOD', keep='last')
    return filtered


def _pit_fill(fin_df, field, trade_dates):
    """Point-in-time前向填充：将季频财务字段按公告日映射到每个交易日。"""
    if fin_df is None or fin_df.empty or field not in fin_df.columns:
        return pd.Series(np.nan, index=trade_dates)
    df = fin_df.copy()
    date_col = 'ACTUAL_ANN_DATE' if 'ACTUAL_ANN_DATE' in df.columns else 'REPORTING_PERIOD'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col, field])
    df = df.sort_values(date_col)
    s = df.set_index(date_col)[field].astype(float)
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return s.reindex(trade_dates, method='ffill')


def _equity_pit(equity_structure, code, field, trade_dates):
    """将股本结构按 CHANGE_DATE 前向填充到交易日。"""
    if equity_structure is None or equity_structure.empty:
        return pd.Series(np.nan, index=trade_dates)
    eq = equity_structure[equity_structure['MARKET_CODE'] == code].copy()
    if eq.empty or field not in eq.columns:
        return pd.Series(np.nan, index=trade_dates)
    eq = eq.sort_values('CHANGE_DATE').drop_duplicates('CHANGE_DATE', keep='last')
    s = eq.set_index('CHANGE_DATE')[field].astype(float)
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return s.reindex(trade_dates, method='ffill')


def _ts_pit(date_index, value_series, trade_dates):
    """通用 point-in-time 前向填充：将任意日期索引+值前向填充到交易日序列。"""
    if hasattr(date_index, 'values'):
        di = date_index.values
    elif isinstance(date_index, (list, tuple)):
        di = np.array(date_index)
    else:
        di = np.asarray(date_index)
    if hasattr(value_series, 'values'):
        vs = value_series.values
    elif isinstance(value_series, (list, tuple)):
        vs = np.array(value_series, dtype=float)
    else:
        vs = np.asarray(value_series, dtype=float)

    if len(di) == 0 or len(vs) == 0:
        return pd.Series(np.nan, index=trade_dates)
    src = pd.Series(vs, index=pd.to_datetime(di))
    src = src[~src.index.duplicated(keep='last')].sort_index()
    return src.reindex(trade_dates, method='ffill')


def _calc_dividend_yield(code, dividend, equity_structure, close, trade_dates, tot_share, mc):
    """计算股息率（日频），point-in-time：每个交易日用截至该日最近一次已实施分红 / 总市值"""
    result = pd.Series(np.nan, index=trade_dates)
    if dividend is None or dividend.empty:
        return result.values
    div_code = dividend[dividend['MARKET_CODE'] == code].copy()
    if div_code.empty or 'DVD_PER_SHARE_PRE_TAX_CASH' not in div_code.columns:
        return result.values

    date_col = 'DATE_DVD_PAYOUT'
    if date_col not in div_code.columns or div_code[date_col].isna().all():
        date_col = 'REPORT_PERIOD'
    if date_col not in div_code.columns:
        return result.values

    div_code = div_code[div_code[date_col].notna() & (div_code[date_col].astype(str) != '')]
    if div_code.empty:
        return result.values

    div_code[date_col] = pd.to_datetime(div_code[date_col])
    div_code = div_code.sort_values(date_col)

    dps_vals = div_code['DVD_PER_SHARE_PRE_TAX_CASH'].astype(float).fillna(0)
    if 'DIV_BASESHARE' in div_code.columns:
        base_shares = div_code['DIV_BASESHARE'].astype(float)
    else:
        base_shares = pd.Series(np.nan, index=div_code.index)

    div_amounts = []
    div_dates_list = []
    for idx in div_code.index:
        dps = dps_vals.loc[idx]
        bs_val = base_shares.loc[idx]
        d = div_code.loc[idx, date_col]
        amount = dps * bs_val * 10000 if pd.notna(bs_val) else np.nan
        div_amounts.append(amount)
        div_dates_list.append(d)

    if not div_dates_list:
        return result.values

    div_pit = _ts_pit(pd.Index(div_dates_list), pd.Series(div_amounts), trade_dates)
    dps_pit = _ts_pit(pd.Index(div_dates_list), dps_vals.reset_index(drop=True), trade_dates)
    div_pit_filled = div_pit.copy()
    na_mask = div_pit.isna() & dps_pit.notna()
    div_pit_filled[na_mask] = dps_pit[na_mask] * tot_share[na_mask] * 10000

    return safe_div(div_pit_filled.values, mc)


def _calc_dividend_yield_ttm(code, dividend, equity_structure, trade_dates, tot_share, mc):
    """计算股息率TTM（日频），滚动12个月内所有已实施分红总额 / 总市值。"""
    result = pd.Series(np.nan, index=trade_dates)
    if dividend is None or dividend.empty:
        return result.values
    div_code = dividend[dividend['MARKET_CODE'] == code].copy()
    if div_code.empty or 'DVD_PER_SHARE_PRE_TAX_CASH' not in div_code.columns:
        return result.values

    date_col = 'DATE_DVD_PAYOUT'
    if date_col not in div_code.columns or div_code[date_col].isna().all():
        date_col = 'REPORT_PERIOD'
    if date_col not in div_code.columns:
        return result.values

    div_code = div_code[div_code[date_col].notna() & (div_code[date_col].astype(str) != '')]
    if div_code.empty:
        return result.values

    div_code[date_col] = pd.to_datetime(div_code[date_col])
    div_code = div_code.sort_values(date_col)
    dps_vals = div_code['DVD_PER_SHARE_PRE_TAX_CASH'].astype(float).fillna(0)

    if 'DIV_BASESHARE' in div_code.columns:
        base_shares = div_code['DIV_BASESHARE'].astype(float)
    else:
        base_shares = pd.Series(np.nan, index=div_code.index)

    div_dates = div_code[date_col].values
    td_dt = trade_dates.to_numpy()

    div_total_daily = np.full(len(trade_dates), np.nan)
    for j in range(len(trade_dates)):
        td = td_dt[j]
        lookback = td - np.timedelta64(365, 'D')
        mask = (div_dates > lookback) & (div_dates <= td)
        if mask.any():
            total = 0.0
            for idx in div_code.index[mask]:
                dps = dps_vals.loc[idx]
                bs_val = base_shares.loc[idx]
                if pd.isna(bs_val):
                    bs_val = tot_share.iloc[j] if pd.notna(tot_share.iloc[j]) else 0
                total += dps * bs_val * 10000
            if total > 0:
                div_total_daily[j] = total

    return safe_div(div_total_daily, mc)


def calc_valuation(code, bs, inc, cf, dividend, kline, equity_structure):
    """计算估值指标（日频时序），返回 DataFrame, index=交易日, 12列。
    市值 = 原始close * 总股本（不复权），财务数据按 point-in-time 前向填充。
    """
    if kline is None or kline.empty:
        return pd.DataFrame()

    kl = kline.copy()
    kl['kline_time'] = pd.to_datetime(kl['kline_time'])
    kl = kl.sort_values('kline_time').drop_duplicates('kline_time', keep='last')
    trade_dates = kl.set_index('kline_time').index
    close = kl.set_index('kline_time')['close'].astype(float)

    tot_share = _equity_pit(equity_structure, code, 'TOT_SHARE', trade_dates)
    total_mkt_cap = close * tot_share * 10000

    bs_f = _filter_statements(bs) if bs is not None and not bs.empty else pd.DataFrame()
    inc_f = _filter_statements(inc) if inc is not None and not inc.empty else pd.DataFrame()
    cf_f = _filter_statements(cf) if cf is not None and not cf.empty else pd.DataFrame()
    for df in [bs_f, inc_f, cf_f]:
        if not df.empty:
            df.sort_values('REPORTING_PERIOD', inplace=True)
            df.drop_duplicates('REPORTING_PERIOD', keep='last', inplace=True)

    ne_total = _pit_fill(bs_f, 'TOT_SHARE_EQUITY_EXCL_MIN_INT', trade_dates)
    oth_eq = _pit_fill(bs_f, 'OTH_EQUITY_TOOLS', trade_dates).fillna(0)
    ne = ne_total - oth_eq
    np_last = _pit_fill(inc_f, 'NET_PRO_EXCL_MIN_INT_INC', trade_dates)
    rev_last = _pit_fill(inc_f, 'OPERA_REV', trade_dates)
    cf_op_last = _pit_fill(cf_f, 'NET_CASH_FLOWS_OPERA_ACT', trade_dates)

    def _ttm_pit(df, field):
        if df is None or df.empty:
            return pd.Series(np.nan, index=trade_dates)
        ttm_s = get_ttm(df.reset_index(drop=True), field)
        tmp = df.copy()
        tmp['_ttm'] = ttm_s.values
        return _pit_fill(tmp, '_ttm', trade_dates)

    np_ttm = _ttm_pit(inc_f, 'NET_PRO_EXCL_MIN_INT_INC')
    rev_ttm = _ttm_pit(inc_f, 'OPERA_REV')
    cf_ttm = _ttm_pit(cf_f, 'NET_CASH_FLOWS_OPERA_ACT')
    fcf_ttm = _ttm_pit(cf_f, 'FREE_CASH_FLOW')
    ncf_ttm = _ttm_pit(cf_f, 'NET_INCR_CASH_AND_CASH_EQU')

    mc = total_mkt_cap.values
    f = pd.DataFrame(index=trade_dates)

    f['市净率'] = safe_div(mc, ne.values)
    f['市现率'] = safe_div(mc, cf_op_last.values)
    f['市盈率'] = safe_div(mc, np_last.values)

    f['股息率'] = _calc_dividend_yield(code, dividend, equity_structure, close, trade_dates, tot_share, mc)

    f['市销率'] = safe_div(mc, rev_last.values)
    f['市现率TTM'] = safe_div(mc, cf_ttm.values)
    f['市盈率TTM'] = safe_div(mc, np_ttm.values)

    f['股息率TTM'] = _calc_dividend_yield_ttm(code, dividend, equity_structure, trade_dates, tot_share, mc)

    f['市销率TTM'] = safe_div(mc, rev_ttm.values)
    f['自由现金流TTM比总市值'] = safe_div(fcf_ttm.values, mc)
    f['净现金流TTM比总市值'] = safe_div(ncf_ttm.values, mc)

    np_ttm_series = np_ttm.copy()
    np_ttm_series.index = pd.to_datetime(np_ttm_series.index)
    prev_dates = np_ttm_series.index - pd.DateOffset(years=1)
    np_ttm_prev_aligned = np_ttm_series.reindex(prev_dates, method='ffill')
    np_ttm_prev_aligned.index = np_ttm_series.index
    growth = safe_div((np_ttm_series - np_ttm_prev_aligned).values,
                      np.abs(np_ttm_prev_aligned.values))
    pe_ttm_vals = f['市盈率TTM'].values
    f['市盈率相对盈利增长率'] = safe_div(pe_ttm_vals, growth * 100)

    return f


# ============================================================
# MCP 数据获取(喂给 calc_valuation 的 skill-schema DataFrame)
# ============================================================

# AmazingData 返回的 MARKET_CODE 形如 "600000.SH":归一化以便任意输入符号都能匹配
_SYMBOL_RE = re.compile(r"^(?:([a-z]{2})?)(\d{6})(?:\.(sh|sz|bj))?$", re.I)


def _norm_code(symbol):
    """把任意股票代码归一为 '600000.SH' 形式;无法解析原样返回。"""
    if not symbol:
        return symbol
    m = _SYMBOL_RE.fullmatch(str(symbol).strip().lower())
    if not m:
        return symbol
    code, exch = m.group(2), m.group(3) or m.group(1)
    if exch:
        return f"{code}.{exch.upper()}"
    if code.startswith(("60", "68", "90")):
        return f"{code}.SH"
    if code.startswith(("00", "30", "20")):
        return f"{code}.SZ"
    if code.startswith(("8", "4", "92")):
        return f"{code}.BJ"
    return symbol


def _pull_records(result, code):
    """从 MCP 返回中解出该标的的记录列表。

    兼容两类返回形状:
      {code: [行...]}                                  (income/cash/equity/dividend)
      {"success":..,"data":{code: [行...]}}            (kline/balance_sheet)
    """
    if not isinstance(result, dict):
        return []
    payload = result
    if isinstance(result.get("data"), dict):
        payload = result["data"]
    rows = payload.get(code)
    if isinstance(rows, list):
        return rows
    # 代码格式不同但 MARKET_CODE 匹配:兜底
    for _k, _v in payload.items():
        if isinstance(_v, list) and _v and isinstance(_v[0], dict) and _v[0].get("MARKET_CODE"):
            if _norm_code(_v[0]["MARKET_CODE"]) == _norm_code(code):
                return _v
    return []


def _df(records, keep=None):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if keep:
        keep = [c for c in keep if c in df.columns]
        df = df[keep]
    # 归一化 MARKET_CODE 列
    if 'MARKET_CODE' in df.columns:
        df['MARKET_CODE'] = df['MARKET_CODE'].apply(lambda s: _norm_code(s) if s is not None else s)
    return df


_VALUATION_COLS = [
    '市净率', '市现率', '市盈率', '股息率', '市销率',
    '市现率TTM', '市盈率TTM', '股息率TTM', '市销率TTM',
    '自由现金流TTM比总市值', '净现金流TTM比总市值', '市盈率相对盈利增长率',
]


def _to_int_date(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(str(value).replace('-', '').replace('/', '').strip())


def _now_int():
    return int(pd.Timestamp.now().strftime('%Y%m%d'))


def _fetch_inputs(code, begin_date, end_date):
    """开一个 MCP 会话,一次性拉齐 kline/三表/股本/分红(关键路径,失败整体抛错)。"""
    norm = _norm_code(code)
    kl_begin, kl_end = _to_int_date(begin_date), _to_int_date(end_date)
    fin_begin, fin_end = 19900101, _now_int()  # 三表拉全历史,保证 get_ttm 有去年年报/去年同期

    async def _run():
        async with mcp_client.open_session() as mcp:
            kres = await mcp_client.get_kline(mcp, [norm], begin_date, end_date, period="day")
            bs_res = await mcp_client.get_balance_sheet(mcp, [norm], fin_begin, fin_end, statement_type="1")
            inc_res = await mcp_client.get_income(mcp, [norm], fin_begin, fin_end, statement_type="1")
            cf_res = await mcp_client.get_cash_flow(mcp, [norm], fin_begin, fin_end, statement_type="1")
            eq_res = await mcp_client.get_equity_structure(
                mcp, [norm], begin_date=fin_begin, end_date=fin_end
            )
            div_res = await mcp_client.get_dividend(
                mcp, [norm], begin_date=fin_begin, end_date=fin_end
            )
            return {
                "kline": kres, "bs": bs_res, "inc": inc_res, "cf": cf_res,
                "equity": eq_res, "dividend": div_res,
            }

    with mcp_client._MCP_LOCK:
        return asyncio.run(_run())


def build_inputs(code, begin_date, end_date):
    """拉取 MCP 数据并组装为 calc_valuation 的 skill-schema DataFrame。

    返回 dict(kline/bs/inc/cf/equity_structure/dividend),kline 为空时其余可为空。
    """
    raw = _fetch_inputs(code, begin_date, end_date)

    krows = _pull_records(raw["kline"], _norm_code(code))
    if not krows:
        raise RuntimeError(f"[valuation] {code} 在 {begin_date}-{end_date} 无K线数据")

    kl = _df(krows)
    if 'trade_time' in kl.columns:
        kl = kl.rename(columns={'trade_time': 'kline_time'})
    if 'kline_time' in kl.columns:
        kl['kline_time'] = pd.to_datetime(kl['kline_time'])
        kl = kl.sort_values('kline_time').drop_duplicates('kline_time', keep='last')
    elif 'date' in kl.columns:
        kl['kline_time'] = pd.to_datetime(kl['date'])

    eq = _df(_pull_records(raw["equity"], _norm_code(code)))
    div = _df(_pull_records(raw["dividend"], _norm_code(code)))
    bs = _df(_pull_records(raw["bs"], _norm_code(code)))
    inc = _df(_pull_records(raw["inc"], _norm_code(code)))
    cf = _df(_pull_records(raw["cf"], _norm_code(code)))

    return {
        "code": _norm_code(code),
        "kline": kl,
        "bs": bs,
        "inc": inc,
        "cf": cf,
        "equity_structure": eq,
        "dividend": div,
    }


def compute_valuation(code, begin_date, end_date) -> pd.DataFrame:
    """计算 600000.SH 等 A股 12 列日频估值时序(2024-06-01/2026-09-25 格式均可)。
    返回 DataFrame, index=交易日, 列 = 12 个估值指标;失败抛 RuntimeError。
    """
    inputs = build_inputs(code, begin_date, end_date)
    if inputs["kline"] is None or inputs["kline"].empty:
        return pd.DataFrame()
    return calc_valuation(
        inputs["code"],
        inputs["bs"],
        inputs["inc"],
        inputs["cf"],
        inputs["dividend"],
        inputs["kline"],
        inputs["equity_structure"],
    )


def _window_percentile(x):
    """窗口内最后一个值(当前值)在该窗口内的百分位(0-100),忽略 NaN。与 zvt ValuationPercentileFactor 同口径。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float((x <= x[-1]).mean() * 100.0)


def compute_valuation_percentile(code, begin_date, end_date, value_cols=None, years=3, min_periods=250):
    """估值时间序列分位现算(0-100,数值越小代表相对自身历史越便宜)。

    口径对齐 zvt `ValuationPercentileFactor`(见 /data/code/zvt/src/zvt/factors/fundamental/
    valuation_percentile_factor.py):日频估值列 rolling(f"{years*365}D", min_periods).apply(窗口内<=当前值的占比)。
    输入 = compute_valuation 的日频列(市净率/市盈率/市盈率TTM),替代 zvt StockValuation 读取。

    begin_date 需比 end_date 早 3 年以上才满足 min_periods=250;返回最新交易日一行:
    {col: {"value": .., "percentile": ..}, "years": years, "source": .., "latest_date": ..}
    """
    df = compute_valuation(code, begin_date, end_date)
    if df.empty:
        return {}
    val_cols = value_cols or ["市净率", "市盈率", "市盈率TTM"]
    window_freq = f"{int(years * 365)}D"

    out = {}
    for col in val_cols:
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        pct = s.rolling(window_freq, min_periods=min_periods).apply(_window_percentile, raw=True)
        latest_val, latest_pct = s.iloc[-1], pct.iloc[-1]
        out[col] = {
            "value": None if pd.isna(latest_val) else round(float(latest_val), 6),
            "percentile": None if pd.isna(latest_pct) else round(float(latest_pct), 2),
        }
    return {"years": years, "source": "valuation计算", "latest_date": str(df.index[-1].date()), "columns": out}


def latest_valuation(code, begin_date, end_date) -> list:
    """最新一期 12 个估值指标,返回 [{name, value}...] 便于 LLM/前端展示。"""
    df = compute_valuation(code, begin_date, end_date)
    if df.empty:
        return []
    latest = df.iloc[-1]
    out = [{"name": "date", "value": str(df.index[-1].date())}]
    for col in _VALUATION_COLS:
        v = latest.get(col, np.nan)
        if pd.isna(v):
            value = None
        else:
            value = None if pd.isna(v) else round(float(v), 6)
        out.append({"name": col, "value": value})
    return out