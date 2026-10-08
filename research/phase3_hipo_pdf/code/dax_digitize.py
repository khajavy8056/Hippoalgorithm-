"""Digitise TradingView-style candles from the Doc/ trade images (pixel units; price = -y).
Light candle = bullish, dark candle = bearish (checked on DAX-2026-05-21 SL chart). Candles are
found on a regular grid (spacing estimated from the ink profile). Drawn entry = blue dotted vertical line;
drawn SL/TP = edges of the pink (risk) / blue (reward) zone boxes."""
import sys, glob, os, json
sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np
from PIL import Image
from numba import njit
from digitize import remove_hlines

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.int32)

def ink_mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    gray = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return (gray < 170) & (sat < 50)

def grid_fit(p, x_lo, x_hi):
    best = (-1, None, None)
    for d in np.arange(9.4, 11.0, 0.01):
        for x0 in np.arange(0, d, 0.25):
            xs = np.round(x0 + np.arange(0, 400) * d).astype(int)
            xs = xs[(xs >= x_lo) & (xs < x_hi)]
            s = p[xs].sum() / max(len(xs), 1)
            if s > best[0]:
                best = (s, d, x0)
    return best[1], best[2]

def digitise(path, y_lo=20, y_hi=760, x_lo=0, x_hi=1820):
    im = load_rgb(path)
    ink = ink_mask(im)
    ink[:, :] = remove_hlines(ink, 25)
    ink[:y_lo] = False; ink[y_hi:] = False
    p = ink[:, x_lo:x_hi].sum(axis=0).astype(float)
    p_full = np.zeros(im.shape[1]); p_full[x_lo:x_hi] = p
    d, x0 = grid_fit(p_full, x_lo + 3, x_hi - 3)
    candles = []
    k = 0
    while True:
        xc = int(round(x0 + k * d))
        if xc >= x_hi - 3:
            break
        win = ink[:, xc - 2: xc + 3]
        cnt = win.sum(axis=1)
        rows = np.where(cnt >= 4)[0]
        if len(rows) >= 2:
            top, bot = rows.min(), rows.max()
            # wick: contiguous ink above/below body in the centre columns
            col = ink[:, xc - 1: xc + 2].any(axis=1)
            y = top - 1
            while y > y_lo and col[y]: y -= 1
            wtop = y + 1
            y = bot + 1
            while y < y_hi and col[y]: y += 1
            wbot = y - 1
            # colour inside body (light vs dark)
            body_px = im[top + 2: bot - 1, xc - 1: xc + 2] if bot - top > 5 else im[top:bot + 1, xc:xc + 1]
            gray = body_px.mean() if body_px.size else 128
            bullish = gray > 180
            candles.append(dict(k=k, xc=xc, top=int(top), bot=int(bot), wtop=int(wtop), wbot=int(wbot),
                                bullish=bool(bullish)))
        k += 1
    return dict(d=float(d), x0=float(x0), candles=candles, shape=im.shape)

def tint_rows(im, xq, cond, y_lo=20, y_hi=760):
    col = im[:, xq]
    m = cond(col)
    ys = np.where(m)[0]
    ys = ys[(ys >= y_lo) & (ys < y_hi)]
    return ys

def drawn_levels(im):
    H, W, _ = im.shape
    # blue dotted entry line: column with many saturated blue pixels
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    blue = (b > 200) & (r < 120) & (g < 160) & ((b - r) > 120)
    colsum = blue[20:760].sum(axis=0)
    xe = int(np.argmax(colsum)) if colsum.max() > 30 else None
    if xe is None:
        return None
    xq = min(xe + 30, W - 30)
    pink = lambda c: (c[:, 0] - c[:, 1] > 10) & (c[:, 0] > 225) & (c[:, 2] - c[:, 1] > 5) & (c[:, 1] < 232)
    blu = lambda c: (c[:, 2] - c[:, 0] > 14) & (c[:, 1] > c[:, 0]) & (c[:, 0] > 200) & (c[:, 0] < 232)
    yp = tint_rows(im, xq, pink)
    yb = tint_rows(im, xq, blu)
    out = dict(xe=xe)
    if len(yp):
        out["pink_top"], out["pink_bot"] = int(yp.min()), int(yp.max())
    if len(yb):
        out["blue_top"], out["blue_bot"] = int(yb.min()), int(yb.max())
    return out

if __name__ == "__main__":
    files = sorted(f for f in glob.glob("/tmp/hipo_doc/DAX-*.png") if "(1)" not in f)
    res = {}
    for f in files:
        name = os.path.basename(f)
        dg = digitise(f)
        im = load_rgb(f)
        lv = drawn_levels(im)
        res[name] = dict(grid_d=dg["d"], grid_x0=dg["x0"], n_candles=len(dg["candles"]), levels=lv,
                         candles=dg["candles"])
        print(f"{name:34s} spacing={dg['d']:.2f} x0={dg['x0']:.2f} candles={len(dg['candles']):3d} "
              f"entry_x={lv.get('xe') if lv else None} pink={lv.get('pink_top') if lv else None}-{lv.get('pink_bot') if lv else None} "
              f"blue={lv.get('blue_top') if lv else None}-{lv.get('blue_bot') if lv else None}")
    json.dump(res, open("/tmp/hipo_r2/dax_digitised.json", "w"))
    print("saved", len(res))
