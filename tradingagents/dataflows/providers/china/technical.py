"""A股技术指标(从 AmazingData skill 算法拆分移植,自包含,无 AmazingData 依赖)。

来源: xysz/ad_skills/ad_technical_analysis/scripts/run_technical_analysis.py
移植要点:
  1. `import AmazingData` 已删除,算子(MA/EMA/SMA/.../SAR)用 pandas/numpy 原生重写,
     语义经 synthetic 探针与 AmazingData 1.1.9 算子逐位对齐(见模型注释)。
  2. 输入:前复权后的 OHLCV DataFrame(列 open/high/low/close/volume/amount,kline_time),
     输出:每个指标的 dict{列名: Series},与 skill 版一致,可比对验收。

算子语义(对齐 AmazingData,min_periods=1 风格):
  MA     = rolling(n, min_periods=1).mean()
  EMA    = ewm(span=n, adjust=False)              (首元素起播)
  SMA    = ewm(alpha=m/n, adjust=False)           (通达信 Y=(m*X+(n-m)*Y')/n)
  MEMA   = ewm(alpha=1/n, adjust=False)           (=SMA(X,N,1))
  EXPMEMA= ewm(span=n, adjust=False), 前 n-1 个置 NaN
  LLV/HHV= rolling(n, min_periods=1).min/max
  REF    = shift(n)
  SUM(x,n): n==0 -> cumsum, else rolling(n, min_periods=1).sum()
  CUMSUM = cumsum
  COUNT  = bool.rolling(n, min_periods=1).sum()
  STD    = rolling(n, min_periods=1).std(ddof=1)
  AVEDEV = rolling(n, min_periods=1).apply(mean(|w-w.mean()|))
  SAR    = 通达信抛物线,首 N 根暖机后出初始极值,方向取"收盘更近哪边极值",
           翻转时新 SAR = 刚结束趋势的极值 ep。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 算子层(自包含)
# ---------------------------------------------------------------------------


def MA(x, n):
    if n <= 0:
        return pd.Series(np.nan, index=x.index)
    return x.rolling(n, min_periods=1).mean()


def EMA(x, n):
    return x.ewm(span=n, adjust=False).mean()


def SMA(x, n, m):
    return x.ewm(alpha=m / n, adjust=False).mean()


def MEMA(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def EXPMEMA(x, n):
    s = x.ewm(span=n, adjust=False).mean()
    if n > 1 and len(s) >= n:
        s.iloc[: n - 1] = np.nan
    return s


def LLV(x, n):
    return x.rolling(n, min_periods=1).min()


def HHV(x, n):
    return x.rolling(n, min_periods=1).max()


def REF(x, n):
    return x.shift(n)


def SUM(x, n):
    # AmazingData 语义:NaN 按 0 参与求和,min_periods=1
    x0 = x.fillna(0)
    if n == 0:
        return x0.cumsum()
    return x0.rolling(n, min_periods=1).sum()


def CUMSUM(x):
    return x.fillna(0).cumsum()


def COUNT(cond, n):
    return cond.fillna(0).rolling(n, min_periods=1).sum()


def STD(x, n):
    return x.rolling(n, min_periods=1).std(ddof=1)


def AVEDEV(x, n):
    return x.rolling(n, min_periods=1).apply(
        lambda w: np.mean(np.abs(w - np.mean(w))), raw=True
    )


def SAR(high, low, close, n, step, max_af):
    """通达信抛物线 SAR,前 N 根暖机,首 N 值 NaN。

    方向:收盘与首 N 根最高/最低哪个更近 → 近最高则空头(初始 SAR=最高),
          近最低则多头(初始 SAR=最低)。af 创新极值递增 step,翻转时新 SAR=刚结束趋势极值。
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    m = len(h)
    out = np.full(m, np.nan)
    if m <= n:
        return pd.Series(out, index=getattr(high, 'index', None))
    i0 = n
    maxh = h[:n].max()
    minl = l[:n].min()
    # 收盘更接近最高 → 空头;否则多头
    short = (maxh - c[n - 1]) <= (c[n - 1] - minl)
    long = not short
    ep = minl if short else maxh
    af = step
    out[i0] = maxh if short else minl
    for i in range(i0 + 1, m):
        if long:
            nxt = out[i - 1] + af * (ep - out[i - 1])
            if h[i] > ep:
                ep = h[i]
                af = min(af + step, max_af)
            if l[i] < nxt:
                long = False
                out[i] = ep
                ep = l[i]
                af = step
            else:
                out[i] = nxt
        else:
            nxt = out[i - 1] + af * (ep - out[i - 1])
            if l[i] < ep:
                ep = l[i]
                af = min(af + step, max_af)
            if h[i] > nxt:
                long = True
                out[i] = ep
                ep = h[i]
                af = step
            else:
                out[i] = nxt
    return pd.Series(out, index=getattr(high, 'index', None))


# ---------------------------------------------------------------------------
# TechnicalIndicators 类(57 个方法,与 skill 版逐字对齐)
# ---------------------------------------------------------------------------


class TechnicalIndicators:
    """常用技术指标。所有方法静态,输入 pandas Series(OHLCV),输出 dict of Series。"""

    # ============ 一、超买超卖型 ============

    @staticmethod
    def KDJ(close, high, low, n=9, m1=3, m2=3):
        llv = LLV(low, n)
        hhv = HHV(high, n)
        denom = hhv - llv
        denom = denom.replace(0, float('nan'))
        rsv = (close - llv) / denom * 100
        k = SMA(rsv, m1, 1)
        d = SMA(k, m2, 1)
        j = 3 * k - 2 * d
        return {'K': k, 'D': d, 'J': j}

    @staticmethod
    def RSI(close, n1=6, n2=12, n3=24):
        lc = REF(close, 1)
        diff = close - lc
        zero = pd.Series(0.0, index=close.index)
        pos_diff = MAX(diff, zero)
        abs_diff = ABS(diff)
        return {
            f'RSI{n1}': SMA(pos_diff, n1, 1) / SMA(abs_diff, n1, 1) * 100,
            f'RSI{n2}': SMA(pos_diff, n2, 1) / SMA(abs_diff, n2, 1) * 100,
            f'RSI{n3}': SMA(pos_diff, n3, 1) / SMA(abs_diff, n3, 1) * 100,
        }

    @staticmethod
    def WR(close, high, low, n1=10, n2=6):
        result = {}
        for n in [n1, n2]:
            hhv = HHV(high, n)
            llv = LLV(low, n)
            denom = hhv - llv
            denom = denom.replace(0, float('nan'))
            result[f'WR{n}'] = (hhv - close) / denom * 100
        return result

    @staticmethod
    def CCI(close, high, low, n=14):
        typ = (high + low + close) / 3
        cci = (typ - MA(typ, n)) * 1000 / (15 * AVEDEV(typ, n))
        return {'CCI': cci}

    @staticmethod
    def ROC(close, n=12, m=6):
        ref_close = REF(close, n)
        roc = (close - ref_close) / ref_close * 100
        maroc = MA(roc, m)
        return {'ROC': roc, 'MAROC': maroc}

    @staticmethod
    def MTM(close, n=12, m=6):
        mtm = close - REF(close, n)
        mamtm = MA(mtm, m)
        return {'MTM': mtm, 'MAMTM': mamtm}

    @staticmethod
    def BIAS(close, n1=6, n2=12, n3=24):
        result = {}
        for n in [n1, n2, n3]:
            ma = MA(close, n)
            result[f'BIAS{n}'] = (close - ma) / ma * 100
        return result

    @staticmethod
    def SKDJ(close, high, low, n=9, m=3):
        lowv = LLV(low, n)
        highv = HHV(high, n)
        denom = highv - lowv
        denom = denom.replace(0, float('nan'))
        rsv = EMA((close - lowv) / denom * 100, m)
        k = EMA(rsv, m)
        d = MA(k, m)
        return {'K': k, 'D': d}

    @staticmethod
    def MFI(close, high, low, volume, n=14, n2=6):
        typ = (high + low + close) / 3
        mr = typ * volume
        ref_typ = REF(typ, 1)
        zero = pd.Series(0.0, index=close.index)
        pmf = SUM(IF(typ > ref_typ, mr, zero), n)
        nmf = SUM(IF(typ < ref_typ, mr, zero), n)
        denom = nmf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + pmf / denom))
        mfi.loc[(pmf > 0) & (nmf == 0)] = 100
        mfi.loc[(pmf == 0) & (nmf == 0)] = 50
        return {'MFI': mfi}

    @staticmethod
    def OSC(close, n=20, m=6):
        osc = (close - MA(close, n)) * 100
        maosc = EMA(osc, m)
        return {'OSC': osc, 'MAOSC': maosc}

    @staticmethod
    def UDL(close, n1=3, n2=5, n3=10, n4=20, m=6):
        udl = (MA(close, n1) + MA(close, n2) + MA(close, n3) + MA(close, n4)) / 4
        maudl = MA(udl, m)
        return {'UDL': udl, 'MAUDL': maudl}

    @staticmethod
    def ACCER(close, n=8):
        x = np.arange(n, dtype=float)
        def calc_slope(window):
            y = np.array(window, dtype=float)
            if len(y) < n or np.isnan(y).any():
                return np.nan
            return np.polyfit(x, y, 1)[0]
        slope_series = close.rolling(window=n, min_periods=n).apply(calc_slope, raw=False)
        accer = slope_series / close
        return {'ACCER': accer}

    @staticmethod
    def RCCD(close, n=59, short=26, long=52, m=26):
        rc = close / REF(close, n)
        arc = SMA(REF(rc, 1), n, 1)
        dif = MA(arc, short) - MA(arc, long)
        rccd = SMA(dif, m, 1)
        return {'DIF': dif, 'RCCD': rccd}

    @staticmethod
    def MARSI(close, m1=10, m2=6):
        lc = REF(close, 1)
        diff = close - lc
        zero = pd.Series(0.0, index=close.index)
        vu = IF(diff >= 0, diff, zero)
        vd = IF(diff < 0, -diff, zero)
        mau1 = MEMA(vu, m1)
        mad1 = MEMA(vd, m1)
        mau2 = MEMA(vu, m2)
        mad2 = MEMA(vd, m2)
        rsi1_raw = 100 * mau1 / (mau1 + mad1)
        rsi2_raw = 100 * mau2 / (mau2 + mad2)
        rsi1 = MA(rsi1_raw, m1)
        rsi2 = MA(rsi2_raw, m2)
        return {'RSI1': rsi1, 'RSI2': rsi2}

    # ============ 二、趋势型 ============

    @staticmethod
    def MACD(close, short=12, long=26, mid=9):
        dif = EMA(close, short) - EMA(close, long)
        dea = EMA(dif, mid)
        macd = 2 * (dif - dea)
        return {'DIF': dif, 'DEA': dea, 'MACD': macd}

    @staticmethod
    def DMI(close, high, low, n=14, m=6):
        ref_high = REF(high, 1)
        ref_low = REF(low, 1)
        ref_close = REF(close, 1)
        zero = pd.Series(0.0, index=close.index)
        tr1 = high - low
        tr2 = ABS(high - ref_close)
        tr3 = ABS(ref_close - low)
        mtr_unit = MAX(MAX(tr1, tr2), tr3)
        mtr = SUM(mtr_unit, n).replace(0, np.nan)
        hd = high - ref_high
        ld = ref_low - low
        dmp_raw = IF((hd > 0) & (hd > ld), hd, zero)
        dmm_raw = IF((ld > 0) & (ld > hd), ld, zero)
        dmp = SUM(dmp_raw, n)
        dmm = SUM(dmm_raw, n)
        pdi = dmp * 100 / mtr
        mdi = dmm * 100 / mtr
        denom = (mdi + pdi).replace(0, np.nan)
        dx = ABS(mdi - pdi) / denom * 100
        adx = MA(dx, m)
        adxr = (adx + REF(adx, m)) / 2
        return {'PDI': pdi, 'MDI': mdi, 'ADX': adx, 'ADXR': adxr}

    @staticmethod
    def DMA(close, n1=10, n2=50, m=10):
        dif = MA(close, n1) - MA(close, n2)
        difma = MA(dif, m)
        return {'DIF': dif, 'AMA': difma}

    @staticmethod
    def TRIX(close, n=12, m=9):
        mtr = EMA(EMA(EMA(close, n), n), n)
        ref_mtr = REF(mtr, 1)
        trix = (mtr - ref_mtr) / ref_mtr * 100
        matrix = MA(trix, m)
        return {'TRIX': trix, 'MATRIX': matrix}

    @staticmethod
    def ARBR(close, open_, high, low, n=26):
        ar = (SUM(high - open_, n) / SUM(open_ - low, n) * 100)
        ref_close = REF(close, 1)
        zero = close * 0
        br = (SUM(MAX(high - ref_close, zero), n) / SUM(MAX(ref_close - low, zero), n) * 100)
        return {'AR': ar, 'BR': br}

    @staticmethod
    def EMV(close, high, low, volume, n=14, m=9):
        vol_ratio = MA(volume, n) / volume
        high_plus_low = high + low
        mid = 100 * (high + low - REF(high_plus_low, 1)) / (high + low)
        hl = high - low
        emv = MA(mid * vol_ratio * hl / MA(hl, n), n)
        maemv = MA(emv, m)
        return {'EMV': emv, 'MAEMV': maemv}

    @staticmethod
    def DPO(close, n=20, m=6):
        ma_close = MA(close, n)
        dpo = close - REF(ma_close, n // 2 + 1)
        madpo = MA(dpo, m)
        return {'DPO': dpo, 'MADPO': madpo}

    @staticmethod
    def VHF(close, n=28):
        hcp = HHV(close, n)
        lcp = LLV(close, n)
        denom = SUM(ABS(close - REF(close, 1)), n)
        denom = denom.replace(0, float('nan'))
        vhf = (hcp - lcp) / denom
        return {'VHF': vhf}

    @staticmethod
    def CHO(close, high, low, volume, n1=10, n2=20, m=6):
        mid = CUMSUM(volume * (2 * close - high - low) / (high + low))
        cho = (MA(mid, n1) - MA(mid, n2)) / 100
        macho = MA(cho, m)
        return {'CHO': cho, 'MACHO': macho}

    @staticmethod
    def DBCD(close, n=5, m=16, t=76):
        ma = MA(close, n)
        bias = (close - ma) / ma
        dif = bias - REF(bias, m)
        dbcd = SMA(dif, t, 1)
        mm = MA(dbcd, 5)
        return {'DBCD': dbcd, 'MM': mm}

    @staticmethod
    def DDI(close, high, low, n=13, n1=26, m=1, m1=5):
        ref_h = REF(high, 1)
        ref_l = REF(low, 1)
        zero = pd.Series(0.0, index=close.index)
        tr = MAX(ABS(high - ref_h), ABS(low - ref_l))
        dmz = IF(high + low <= ref_h + ref_l, zero, tr)
        dmf = IF(high + low >= ref_h + ref_l, zero, tr)
        sum_dmz = SUM(dmz, n)
        sum_dmf = SUM(dmf, n)
        denom = (sum_dmz + sum_dmf).replace(0, np.nan)
        diz = sum_dmz / denom
        dif = sum_dmf / denom
        ddi = diz - dif
        addi = SMA(ddi, n1, m)
        ad_line = MA(addi, m1)
        return {'DDI': ddi, 'ADDI': addi, 'ADL': ad_line}

    @staticmethod
    def JS(close, n=5, m1=5, m2=10, m3=20):
        ref_close = REF(close, n)
        js = (close - ref_close) / (n * ref_close) * 100
        return {
            'JS': js,
            f'MAJ{m1}': MA(js, m1),
            f'MAJ{m2}': MA(js, m2),
            f'MAJ{m3}': MA(js, m3),
        }

    @staticmethod
    def QACD(close, n1=12, n2=26, m=9):
        dif = EMA(close, n1) - EMA(close, n2)
        macd = EMA(dif, m)
        ddif = dif - macd
        return {'DIF': dif, 'MACD': macd, 'DDIF': ddif}

    @staticmethod
    def UOS(close, high, low, n1=7, n2=14, n3=28, m=6):
        ref_c = REF(close, 1)
        th = MAX(high, ref_c)
        tl = MIN(low, ref_c)
        acc1 = SUM(close - tl, n1) / SUM(th - tl, n1)
        acc2 = SUM(close - tl, n2) / SUM(th - tl, n2)
        acc3 = SUM(close - tl, n3) / SUM(th - tl, n3)
        uos = (acc1 * n2 * n3 + acc2 * n1 * n3 + acc3 * n1 * n2) * 100 / (n1 * n2 + n1 * n3 + n2 * n3)
        mauos = EMA(uos, m)
        return {'UOS': uos, 'MAUOS': mauos}

    # ============ 三、能量型 ============

    @staticmethod
    def CR(close, high, low, n=26, m1=10, m2=20, m3=40, m4=62):
        mid = REF(high + low, 1) / 2
        zero = close * 0
        up = MAX(high - mid, zero)
        down = MAX(mid - low, zero)
        down_sum = SUM(down, n).replace(0, np.nan)
        cr = SUM(up, n) / down_sum * 100
        ma1 = REF(MA(cr, m1), int(m1 / 2.5 + 1))
        ma2 = REF(MA(cr, m2), int(m2 / 2.5 + 1))
        ma3 = REF(MA(cr, m3), int(m3 / 2.5 + 1))
        ma4 = REF(MA(cr, m4), int(m4 / 2.5 + 1))
        return {'CR': cr, f'MA{m1}': ma1, f'MA{m2}': ma2, f'MA{m3}': ma3, f'MA{m4}': ma4}

    @staticmethod
    def PSY(close, n=12, m=6):
        cond = close > REF(close, 1)
        psy = COUNT(cond, n) / n * 100
        mapsy = MA(psy, m)
        return {'PSY': psy, 'PSYMA': mapsy}

    @staticmethod
    def MASS(high, low, n1=9, n2=25, m=6):
        hl_ema = MA(high - low, n1)
        mass = SUM(hl_ema / MA(hl_ema, n1), n2)
        mamass = MA(mass, m)
        return {'MASS': mass, 'MAMASS': mamass}

    @staticmethod
    def PCNT(close, m=5):
        ref_close = REF(close, 1)
        pcnt = (close - ref_close) / close * 100
        mapcnt = EXPMEMA(pcnt, m)
        return {'PCNT': pcnt, 'MAPCNT': mapcnt}

    @staticmethod
    def WAD(close, high, low, m=30):
        ref_c = REF(close, 1)
        mida = close - MIN(low, ref_c)
        midb = IF(close < ref_c, close - MAX(ref_c, high), 0)
        wad = SUM(IF(close > ref_c, mida, midb), 0)
        mawad = MA(wad, m)
        return {'WAD': wad, 'MAWAD': mawad}

    # ============ 四、成交量型 ============

    @staticmethod
    def OBV(close, volume, m=30):
        ref_close = REF(close, 1)
        direction = SIGN(close - ref_close).fillna(0)
        obv = CUMSUM(direction * volume)
        if len(obv) > 0 and len(volume) > 0:
            obv.iloc[0] = volume.iloc[0]
        maobv = MA(obv, m)
        return {'OBV': obv, 'MAOBV': maobv}

    @staticmethod
    def VR(close, volume, n=26, m=6):
        ref_close = REF(close, 1)
        zero = pd.Series(0.0, index=close.index)
        av = SUM(IF(close > ref_close, volume, zero), n)
        bv = SUM(IF(close < ref_close, volume, zero), n)
        cv = SUM(IF(close == ref_close, volume, zero), n)
        vr = (av + cv / 2) / (bv + cv / 2) * 100
        mavr = MA(vr, m)
        return {'VR': vr, 'MAVR': mavr}

    @staticmethod
    def VOLMA(volume, n1=5, n2=10):
        return {f'VOLMA{n1}': MA(volume, n1), f'VOLMA{n2}': MA(volume, n2)}

    @staticmethod
    def WVAD(close, open_, high, low, volume, n=24, m=6):
        wvad = SUM((close - open_) / (high - low) * volume, n) / 10000
        mawvad = MA(wvad, m)
        return {'WVAD': wvad, 'MAWVAD': mawvad}

    @staticmethod
    def VOSC(volume, short=12, long=26):
        ma_short = MA(volume, short)
        ma_long = MA(volume, long)
        vosc = (ma_short - ma_long) / ma_short * 100
        return {'VOSC': vosc}

    @staticmethod
    def VRSI(volume, n1=6, n2=12, n3=24):
        lv = REF(volume, 1)
        diff = volume - lv
        zero = pd.Series(0.0, index=volume.index)
        pos_diff = MAX(diff, zero)
        abs_diff = ABS(diff)
        result = {}
        for n in [n1, n2, n3]:
            result[f'VRSI{n}'] = SMA(pos_diff, n, 1) / SMA(abs_diff, n, 1) * 100
        return result

    @staticmethod
    def VSTD(volume, n=10):
        return {'VSTD': STD(volume, n)}

    @staticmethod
    def AMO(amount, n1=5, n2=10):
        return {
            'AMOW': amount / 10000,
            f'AMO{n1}': MA(amount / 10000, n1),
            f'AMO{n2}': MA(amount / 10000, n2),
        }

    @staticmethod
    def TAPI(close, amount, n=6):
        tapi = amount / close
        matapi = MA(tapi, n)
        return {'TAPI': tapi, 'MATAPI': matapi}

    # ============ 五、均线型 ============

    @staticmethod
    def MA(close, m1=5, m2=10, m3=20, m4=60, m5=0, m6=0, m7=0, m8=0):
        return {
            f'MA{m1}': MA(close, m1),
            f'MA{m2}': MA(close, m2),
            f'MA{m3}': MA(close, m3),
            f'MA{m4}': MA(close, m4),
            f'MA{m5}': MA(close, m5),
            f'MA{m6}': MA(close, m6),
            f'MA{m7}': MA(close, m7),
            f'MA{m8}': MA(close, m8),
        }

    @staticmethod
    def EXPMA(close, n1=12, n2=50):
        return {f'EXPMA{n1}': EMA(close, n1), f'EXPMA{n2}': EMA(close, n2)}

    @staticmethod
    def BBI(close, m1=3, m2=6, m3=12, m4=24):
        bbi = (MA(close, m1) + MA(close, m2) + MA(close, m3) + MA(close, m4)) / 4
        return {'BBI': bbi}

    @staticmethod
    def AMV(volume, amount, n1=5, n2=13, n3=34, n4=60):
        return {
            f'AMV{n1}': SUM(amount, n1) / SUM(volume, n1),
            f'AMV{n2}': SUM(amount, n2) / SUM(volume, n2),
            f'AMV{n3}': SUM(amount, n3) / SUM(volume, n3),
            f'AMV{n4}': SUM(amount, n4) / SUM(volume, n4),
        }

    # ============ 六、路径型 ============

    @staticmethod
    def BOLL(close, n=20, k=2):
        mid = MA(close, n)
        vart1 = POW((close - mid), 2)
        vart2 = MA(vart1, n)
        vart3 = SQRT(vart2)
        upper = mid + k * vart3
        lower = mid - k * vart3
        boll = REF(mid, 1)
        ub = REF(upper, 1)
        lb = REF(lower, 1)
        return {'BOLL': boll, 'UB': ub, 'LB': lb}

    @staticmethod
    def ENE(close, n=25, m1=6, m2=6):
        ma = MA(close, n)
        upper = ma * (1 + m1 / 100)
        lower = ma * (1 - m2 / 100)
        ene = (upper + lower) / 2
        return {'UPPER': upper, 'ENE': ene, 'LOWER': lower}

    @staticmethod
    def MIKE(close, high, low, n=10):
        hlc = REF(MA((high + low + close) / 3, n), 1)
        hv = EMA(HHV(high, n), 3)
        lv = EMA(LLV(low, n), 3)
        wr = EMA(hlc * 2 - lv, 3)
        mr = EMA(hlc + hv - lv, 3)
        sr = EMA(2 * hv - lv, 3)
        ws = EMA(hlc * 2 - hv, 3)
        ms = EMA(hlc - hv + lv, 3)
        ss = EMA(2 * lv - hv, 3)
        return {'WEKR': wr, 'MIDR': mr, 'STOR': sr, 'WEKS': ws, 'MIDS': ms, 'STOS': ss}

    @staticmethod
    def PBX(close, m1=4, m2=6, m3=9, m4=13, m5=18, m6=24):
        return {
            f'PBX{m1}': (EMA(close, m1) + EMA(close, m1 * 2) + EMA(close, m1 * 4)) / 3,
            f'PBX{m2}': (EMA(close, m2) + EMA(close, m2 * 2) + EMA(close, m2 * 4)) / 3,
            f'PBX{m3}': (EMA(close, m3) + EMA(close, m3 * 2) + EMA(close, m3 * 4)) / 3,
            f'PBX{m4}': (EMA(close, m4) + EMA(close, m4 * 2) + EMA(close, m4 * 4)) / 3,
            f'PBX{m5}': (EMA(close, m5) + EMA(close, m5 * 2) + EMA(close, m5 * 4)) / 3,
            f'PBX{m6}': (EMA(close, m6) + EMA(close, m6 * 2) + EMA(close, m6 * 4)) / 3,
        }

    @staticmethod
    def XS(close, high, low, n=13):
        # 与 skill 版逐字一致:volume 未入参,保持原样(上游 skill 同样会 NameError)
        var2 = close * volume
        p1 = EMA(var2, 3) / EMA(volume, 3)
        p2 = EMA(var2, 6) / EMA(volume, 6)
        p3 = EMA(var2, 12) / EMA(volume, 12)
        p4 = EMA(var2, 24) / EMA(volume, 24)
        var3 = EMA((p1 + p2 + p3 + p4) / 4, n)
        sup = 1.06 * var3
        sdn = var3 * 0.94
        var4 = EMA(close, 9)
        lup = EMA(var4 * 1.14, 5)
        ldn = EMA(var4 * 0.86, 5)
        return {'SUP': sup, 'SDN': sdn, 'LUP': lup, 'LDN': ldn}

    @staticmethod
    def BBIBOLL(close, n=11, m=6):
        bbi = (MA(close, 3) + MA(close, 6) + MA(close, 12) + MA(close, 24)) / 4
        std = STD(bbi, n)
        upper = bbi + m * std
        lower = bbi - m * std
        return {'BBIBOLL': bbi, 'UPPER': upper, 'LOWER': lower}

    # ============ 七、其他型 ============

    @staticmethod
    def ASI(close, open_, high, low, m1=26, m2=10):
        ref_c = REF(close, 1)
        ref_o = REF(open_, 1)
        ref_l = REF(low, 1)
        aa = ABS(high - ref_c)
        bb = ABS(low - ref_c)
        cc = ABS(high - ref_l)
        dd = ABS(ref_c - ref_o)
        r_a = aa + bb / 2 + dd / 4
        r_b = bb + aa / 2 + dd / 4
        r_c = cc + dd / 4
        r = IF((aa > bb) & (aa > cc), r_a, IF((bb > cc) & (bb > aa), r_b, r_c))
        r = r.replace(0, np.nan)
        x = (close - ref_c + (close - open_) / 2 + ref_c - ref_o)
        si = 16 * x / r * MAX(aa, bb)
        asi = SUM(si, m1)
        asit = MA(asi, m2)
        return {'SI': si, 'ASI': asi}

    @staticmethod
    def ATR(close, high, low, n=14):
        mtr = MAX(
            MAX((high - low), ABS(REF(close, 1) - high)),
            ABS(REF(close, 1) - low),
        )
        atr = MA(mtr, n)
        return {'MTR': mtr, 'ATR': atr}

    @staticmethod
    def SAR(close, high, low, n=4, step=0.02, max_af=0.2):
        sar = SAR(high, low, close, n, step, max_af)
        return {'SAR': sar}

    @staticmethod
    def CDP(close, high, low):
        ref_h = REF(high, 1)
        ref_l = REF(low, 1)
        ref_c = REF(close, 1)
        cdp = (ref_h + ref_l + ref_c) / 3
        ah = 2 * cdp + ref_h - 2 * ref_l
        nh = 2 * cdp - ref_l
        nl = 2 * cdp - ref_h
        al = 2 * cdp - 2 * ref_h + ref_l
        return {'AH': ah, 'NH': nh, 'CDP': cdp, 'NL': nl, 'AL': al}


# 同名工具函数参数转发(供静态方法内部按名字取算子)
def POW(x, p):
    return x ** p


def SQRT(x):
    return np.sqrt(x)


def ABS(x):
    return np.abs(x)


def SIGN(x):
    return np.sign(x)


def MAX(a, b):
    return np.maximum(a, b)


def MIN(a, b):
    return np.minimum(a, b)


def IF(cond, a, b):
    return pd.Series(np.where(cond, a, b), index=getattr(cond, 'index', None))


# ---------------------------------------------------------------------------
# 分类注册 + 计算入口
# ---------------------------------------------------------------------------

ALL_INDICATORS = {
    "overbought_oversold": ("一、超买超卖型", [
        ("KDJ", lambda TI, p: TI.KDJ(p.close, p.high, p.low)),
        ("RSI", lambda TI, p: TI.RSI(p.close)),
        ("WR", lambda TI, p: TI.WR(p.close, p.high, p.low)),
        ("CCI", lambda TI, p: TI.CCI(p.close, p.high, p.low)),
        ("ROC", lambda TI, p: TI.ROC(p.close)),
        ("MTM", lambda TI, p: TI.MTM(p.close)),
        ("BIAS", lambda TI, p: TI.BIAS(p.close)),
        ("SKDJ", lambda TI, p: TI.SKDJ(p.close, p.high, p.low)),
        ("MFI", lambda TI, p: TI.MFI(p.close, p.high, p.low, p.volume)),
        ("OSC", lambda TI, p: TI.OSC(p.close)),
        ("UDL", lambda TI, p: TI.UDL(p.close)),
        ("ACCER", lambda TI, p: TI.ACCER(p.close)),
        ("RCCD", lambda TI, p: TI.RCCD(p.close)),
        ("MARSI", lambda TI, p: TI.MARSI(p.close)),
    ]),
    "trend": ("二、趋势型", [
        ("MACD", lambda TI, p: TI.MACD(p.close)),
        ("DMI", lambda TI, p: TI.DMI(p.close, p.high, p.low)),
        ("DMA", lambda TI, p: TI.DMA(p.close)),
        ("TRIX", lambda TI, p: TI.TRIX(p.close)),
        ("ARBR", lambda TI, p: TI.ARBR(p.close, p.open_, p.high, p.low)),
        ("EMV", lambda TI, p: TI.EMV(p.close, p.high, p.low, p.volume)),
        ("DPO", lambda TI, p: TI.DPO(p.close)),
        ("VHF", lambda TI, p: TI.VHF(p.close)),
        ("CHO", lambda TI, p: TI.CHO(p.close, p.high, p.low, p.volume)),
        ("DBCD", lambda TI, p: TI.DBCD(p.close)),
        ("DDI", lambda TI, p: TI.DDI(p.close, p.high, p.low)),
        ("JS", lambda TI, p: TI.JS(p.close)),
        ("QACD", lambda TI, p: TI.QACD(p.close)),
        ("UOS", lambda TI, p: TI.UOS(p.close, p.high, p.low)),
    ]),
    "energy": ("三、能量型", [
        ("CR", lambda TI, p: TI.CR(p.close, p.high, p.low)),
        ("PSY", lambda TI, p: TI.PSY(p.close)),
        ("MASS", lambda TI, p: TI.MASS(p.high, p.low)),
        ("PCNT", lambda TI, p: TI.PCNT(p.close)),
        ("WAD", lambda TI, p: TI.WAD(p.close, p.high, p.low)),
    ]),
    "volume": ("四、成交量型", [
        ("OBV", lambda TI, p: TI.OBV(p.close, p.volume)),
        ("VR", lambda TI, p: TI.VR(p.close, p.volume)),
        ("VOLMA", lambda TI, p: TI.VOLMA(p.volume)),
        ("WVAD", lambda TI, p: TI.WVAD(p.close, p.open_, p.high, p.low, p.volume)),
        ("VOSC", lambda TI, p: TI.VOSC(p.volume)),
        ("VRSI", lambda TI, p: TI.VRSI(p.volume)),
        ("VSTD", lambda TI, p: TI.VSTD(p.volume)),
        ("AMO", lambda TI, p: TI.AMO(p.amount)),
        ("TAPI", lambda TI, p: TI.TAPI(p.close, p.amount)),
    ]),
    "ma": ("五、均线型", [
        ("MA", lambda TI, p: TI.MA(p.close)),
        ("EXPMA", lambda TI, p: TI.EXPMA(p.close)),
        ("BBI", lambda TI, p: TI.BBI(p.close)),
        ("AMV", lambda TI, p: TI.AMV(p.volume, p.amount)),
    ]),
    "path": ("六、路径型", [
        ("BOLL", lambda TI, p: TI.BOLL(p.close)),
        ("ENE", lambda TI, p: TI.ENE(p.close)),
        ("MIKE", lambda TI, p: TI.MIKE(p.close, p.high, p.low)),
        ("PBX", lambda TI, p: TI.PBX(p.close)),
        ("XS", lambda TI, p: TI.XS(p.close, p.high, p.low)),
        ("BBIBOLL", lambda TI, p: TI.BBIBOLL(p.close)),
    ]),
    "other": ("七、其他型", [
        ("ASI", lambda TI, p: TI.ASI(p.close, p.open_, p.high, p.low)),
        ("ATR", lambda TI, p: TI.ATR(p.close, p.high, p.low)),
        ("SAR", lambda TI, p: TI.SAR(p.close, p.high, p.low)),
        ("CDP", lambda TI, p: TI.CDP(p.close, p.high, p.low)),
    ]),
}

CATEGORY_NAMES = [c for c, (_, lst) in ALL_INDICATORS.items() for _ in (lst,)]


def is_stock(code):
    """判断代码是否为股票(与 skill 一致:6位数字,SH首位6,SZ首位0/3)。"""
    parts = code.split('.')
    pure_code, market = parts[0], parts[1].upper()
    if len(pure_code) != 6 or not pure_code.isdigit():
        return False
    if market == 'SH' and pure_code[0] == '6':
        return True
    if market == 'SZ' and pure_code[0] in ('0', '3'):
        return True
    return False


def forward_adjust(df, backward_factor, code):
    """前复权(只调 OHLC 价格),与 skill 版一致。backward_factor: Series(index=date, value=factor)。"""
    df_adj = df.copy()
    factor = backward_factor
    if factor is None or len(factor) == 0:
        return df_adj
    if 'kline_time' in df_adj.columns:
        kline_dates = pd.to_datetime(df_adj['kline_time'])
    else:
        kline_dates = pd.to_datetime(df_adj.index)
    factor_aligned = factor.reindex(kline_dates, method='ffill')
    factor_aligned = factor_aligned.values
    latest_factor = factor_aligned[~pd.isna(factor_aligned)]
    if len(latest_factor) == 0:
        return df_adj
    latest_factor = latest_factor[-1]
    adj_ratio = factor_aligned / latest_factor
    for col in ['open', 'high', 'low', 'close']:
        if col in df_adj.columns:
            df_adj[col] = df_adj[col] * adj_ratio
    return df_adj


def _last_value(value):
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if abs(value) >= 100:
        return round(float(value), 2)
    if abs(value) >= 10:
        return round(float(value), 3)
    if abs(value) >= 1:
        return round(float(value), 4)
    return round(float(value), 6)


def _calc_df(kline):
    """从 kline DataFrame 抽 OHLCV Series 命名空间。"""
    class P:
        pass
    p = P()
    close_s = kline['close'].astype(float) if 'close' in kline.columns else pd.Series(np.nan, index=kline.index)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in kline.columns:
            setattr(p, col, kline[col].astype(float))
        else:
            setattr(p, col, pd.Series(np.nan, index=kline.index))
    if 'amount' in kline.columns:
        p.amount = kline['amount'].astype(float)
    else:
        p.amount = close_s * p.volume
    p.open_ = p.open
    return p


def compute_indicators(kline, indicator=None, category=None):
    """计算技术指标,返回按类别分组的 {ind_name: {key: last_value}}。
    kline: 前复权 OHLCV DataFrame(需含 kline_time)。与 skill 版对比验收用。
    """
    TI = TechnicalIndicators
    p = _calc_df(kline)

    if indicator:
        indicator = indicator.upper()
        for cat_key, (_, inds) in ALL_INDICATORS.items():
            for ind_name, fn in inds:
                if ind_name.upper() == indicator:
                    result = fn(TI, p)
                    values = {k: _last_value(s.iloc[-1]) for k, s in result.items()}
                    return {"category": cat_key, "name": ind_name, "values": values}
        raise ValueError(f"未找到指标: {indicator}")

    if category and category not in ALL_INDICATORS:
        raise ValueError(f"未找到类别: {category}")

    cats = {category: ALL_INDICATORS[category]} if category else ALL_INDICATORS
    out = {}
    for cat_key, (cat_name, inds) in cats.items():
        cat_results = []
        for ind_name, fn in inds:
            try:
                result = fn(TI, p)
                values = {k: _last_value(s.iloc[-1]) for k, s in result.items()}
                cat_results.append({"name": ind_name, "values": values})
            except Exception as e:
                cat_results.append({"name": ind_name, "values": {}, "error": f"计算失败: {e}"})
        out[cat_key] = {"name": cat_name, "indicators": cat_results}
    return out