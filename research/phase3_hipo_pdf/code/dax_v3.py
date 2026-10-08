import sys, glob, os
sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np, pandas as pd
from dax_digitize import load_rgb, ink_mask, grid_fit
from hipo_core import scan_direction, params_vector, atr_wilder
from digitize import remove_hlines

def runs_of(mask1d):
    idx = np.where(mask1d)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]; ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts, ends))

def digitise_v3(im, y_lo=20, y_hi=745, x_lo=0, x_hi=1800, gap=150):
    ink = ink_mask(im)
    ink = remove_hlines(ink, 25)
    ink[:y_lo] = False; ink[y_hi:] = False
    p = np.zeros(im.shape[1]); p[x_lo:x_hi] = ink[:, x_lo:x_hi].sum(axis=0)
    d, x0 = grid_fit(p, x_lo + 3, x_hi - 3)
    cands = []
    k = 0
    while True:
        xc = int(round(x0 + k * d))
        if xc >= x_hi - 3:
            break
        win = ink[:, xc - 2: xc + 3]
        cnt = win.sum(axis=1)
        good = np.where(cnt >= 4)[0]
        if len(good) >= 2:
            # cluster good rows (outline rows of light candles / whole body of dark candles)
            clusters = [[good[0]]]
            for g in good[1:]:
                if g - clusters[-1][-1] <= gap:
                    clusters[-1].append(g)
                else:
                    clusters.append([g])
            cl = max(clusters, key=lambda c: (len(c), -abs(np.mean(c) - 370)))
            top, bot = int(min(cl)), int(max(cl))
            if bot - top >= 1 or len(cl) >= 2:
                col = ink[:, xc - 1: xc + 2].any(axis=1)
                y = top - 1
                while y > y_lo and col[y]: y -= 1
                wtop = y + 1
                y = bot + 1
                while y < y_hi and col[y]: y += 1
                wbot = y - 1
                if bot - top >= 3:
                    samp = im[top + 1: bot, xc - 1: xc + 2]
                else:
                    samp = im[top: bot + 1, xc - 2: xc + 3]
                gray = float(samp.mean())
                cands.append(dict(k=k, xc=xc, top=top, bot=bot, wtop=wtop, wbot=wbot, bullish=bool(gray > 180), gray=gray))
        k += 1
    return dict(d=float(d), x0=float(x0), candles=cands)

def zone_masks(im):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    pink = (R - G >= 14) & (R >= 225) & (B - G >= 0)
    blue = (B - R >= 14) & (R >= 190) & (G >= R - 2) & (B >= 230)
    return pink, blue

def zones(im):
    pink, blue = zone_masks(im)
    out = {}
    for nm, m in (("pink", pink), ("blue", blue)):
        m = m.copy(); m[:20] = False; m[760:] = False
        colcnt = m.sum(axis=0)
        xs = np.where(colcnt >= 25)[0]
        if len(xs) == 0:
            out[nm] = None; continue
        xl = int(xs.min())
        band = m[:, xl + 4: xl + 40].mean(axis=1) >= 0.5
        ys = np.where(band)[0]
        out[nm] = dict(xl=xl, top=int(ys.min()), bot=int(ys.max())) if len(ys) else None
    return out

def drawn_from_zones(im):
    z = zones(im)
    zp, zb = z["pink"], z["blue"]
    if zp is None and zb is None:
        return None
    xl = min([v["xl"] for v in (zp, zb) if v is not None])
    if zp is not None and zb is not None:
        short = (zp["top"] + zp["bot"]) / 2 < (zb["top"] + zb["bot"]) / 2
        if short:
            return dict(dir=-1, xl=xl, entry_y=zp["bot"], sl_y=zp["top"], tp_y=zb["bot"], R_px=zp["bot"] - zp["top"], zp=zp, zb=zb)
        return dict(dir=+1, xl=xl, entry_y=zp["top"], sl_y=zp["bot"], tp_y=zb["top"], R_px=zp["bot"] - zp["top"], zp=zp, zb=zb)
    return dict(dir=None, xl=xl, zp=zp, zb=zb)

def run_image(path):
    im = load_rgb(path)
    dg = digitise_v3(im)
    cands = dg["candles"]
    n = len(cands)
    xs = np.array([c["xc"] for c in cands], float)
    O = np.zeros(n); Hh = np.zeros(n); Ll = np.zeros(n); Cc = np.zeros(n)
    for i, c in enumerate(cands):
        if c["bullish"]:
            O[i], Cc[i] = -c["bot"], -c["top"]
        else:
            O[i], Cc[i] = -c["top"], -c["bot"]
        Hh[i], Ll[i] = -c["wtop"], -c["wbot"]
    ATR = atr_wilder(Hh, Ll, Cc, 14)
    prm = params_vector()
    sigs = []
    for direction in (+1, -1):
        res = scan_direction(O, Hh, Ll, Cc, ATR, prm, direction)
        for row in res:
            ei = int(row[0])
            sigs.append(dict(dir=-1 if direction > 0 else +1, e_idx=ei, x=float(xs[ei]), entry=float(row[5]),
                             sl=float(row[6]), tp=float(row[7]), atr=float(ATR[ei]), ab=float(row[8]),
                             bc_bars=int(row[11]), ab_bars=int(row[12]), tdiv=int(row[13])))
    return dict(d=dg["d"], n=n, xs=xs, O=O, H=Hh, L=Ll, C=Cc, ATR=ATR, sigs=sigs, drawn=drawn_from_zones(im), cands=cands)

if __name__ == "__main__":
    files = sorted(f for f in glob.glob("/tmp/hipo_doc/DAX-*.png") if "(1)" not in f)
    rows = []
    for f in files:
        name = os.path.basename(f)
        o = run_image(f)
        dt = o["drawn"]; d = o["d"]
        dsc = "none"
        matched = None; sl_err = tp_err = np.nan; rr = np.nan
        if dt is not None and dt.get("dir") is not None:
            dsc = "SHORT" if dt["dir"] < 0 else "LONG"
            cand = [s for s in o["sigs"] if s["dir"] == dt["dir"] and abs(s["x"] - dt["xl"]) <= 0.7 * d]
            matched = len(cand) > 0
            if cand:
                s = min(cand, key=lambda s: abs(s["x"] - dt["xl"]))
                sl_d = -dt["sl_y"]; tp_d = -dt["tp_y"]
                sl_err = (s["sl"] - sl_d) / s["atr"]; tp_err = (s["tp"] - tp_d) / s["atr"]
                rr = abs(s["entry"] - s["sl"]) / max(dt["R_px"], 1e-9)
        rows.append(dict(image=name, drawn=dsc, drawn_x=None if dt is None or dt.get("xl") is None else dt["xl"],
                         drawn_R_px=None if dt is None or dt.get("R_px") is None else dt["R_px"],
                         n_candles=o["n"], spacing=round(d, 2), detector_signals=len(o["sigs"]),
                         det_same_dir_at_entry=matched, sl_err_atr=None if np.isnan(sl_err) else round(float(sl_err), 3),
                         tp_err_atr=None if np.isnan(tp_err) else round(float(tp_err), 3),
                         R_ratio=None if np.isnan(rr) else round(float(rr), 3)))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    df.to_csv("/tmp/hipo_r2/dax_compare_v3.csv", index=False)
