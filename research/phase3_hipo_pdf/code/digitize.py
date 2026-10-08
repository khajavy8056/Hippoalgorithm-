import numpy as np
from PIL import Image

def clusters(idx):
    out = []
    if len(idx) == 0:
        return out
    s = p = idx[0]
    for v in idx[1:]:
        if v > p + 1:
            out.append((s, p)); s = v
        p = v
    out.append((s, p))
    return out

def load(path):
    return np.array(Image.open(path).convert("RGB")).astype(int)

def green_mask(im):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    return (G - R > 60) & (B - R > 40) & (G > 110)

def red_mask(im):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    return (R - G > 120) & (R - B > 120) & (R > 180)

def candle_columns(mask, x0, x1, y0, y1, min_pixels=3):
    """return x-centres of columns that hold candle-coloured pixels (clustered)"""
    sub = mask[y0:y1, x0:x1]
    colcount = sub.sum(axis=0)
    xs = np.where(colcount >= min_pixels)[0] + x0
    return clusters(xs)

def extract(im, xc, gmask, rmask, y0, y1):
    """OHLC in pixel-y for one candle centred at column xc"""
    g = gmask[y0:y1]; r = rmask[y0:y1]
    m = g | r
    center = m[:, xc]
    if center.sum() == 0:
        return None
    ys_wick = np.where(center)[0] + y0
    # body: rows where a horizontal run of >=4 px of the same colour exists in xc-2..xc+2
    win = m[:, xc-2:xc+3]
    body_rows = np.where(win.sum(axis=1) >= 4)[0] + y0
    colour = 'G' if g[:, xc].sum() >= r[:, xc].sum() else 'R'
    res = dict(xc=xc, colour=colour, wick_top=ys_wick.min(), wick_bot=ys_wick.max())
    if len(body_rows):
        res['body_top'] = body_rows.min(); res['body_bot'] = body_rows.max()
    else:
        res['body_top'] = res['body_bot'] = (ys_wick.min() + ys_wick.max()) / 2
    return res

def make_price(y_ref_px, val_ref, px_per_unit):
    """price = val_ref - (y - y_ref_px)/px_per_unit (y grows downwards)"""
    return lambda y: val_ref - (y - y_ref_px) / px_per_unit

def remove_hlines(mask, minrun=15):
    """drop horizontal drawing lines: pixels that belong to a horizontal run >= minrun"""
    out = mask.copy()
    H, W = mask.shape
    for y in range(H):
        row = mask[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(int), [0]]))
        starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= minrun:
                out[y, s:e] = False
    return out
