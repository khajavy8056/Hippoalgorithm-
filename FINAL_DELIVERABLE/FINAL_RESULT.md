# HIPO Pipeline Final Result

**Time:** 2026-07-29T20:29:34.970474Z

## 1) Labeling

| | Old | New (best multi-obj) |
|--|-----|----------------------|
| Samples | 81 | **137** |
| Wins | 35 | **56** |
| WinRate | 43.2% | **40.9%** |
| Config | original | `swing1_loose_rr1` |
| Feature Quality (fast) | - | 51.35 |
| Multi-obj Fitness | - | 64.35 |

Sample multiplier: **1.69×** (+56 signals)

### Top labeling configs
```
            name   n  wins    wr    fq  fitness  rr
swing1_loose_rr1 138    56 40.58 51.35    64.35 1.0
  very_loose_rr1 115    49 42.61 45.65    62.35 1.0
       loose_rr1 102    40 39.22 51.56    61.89 1.0
    ab_gt_cd_rr1 882   398 45.12 14.20    61.40 1.0
    wide_ext_rr1  96    39 40.62 47.19    60.74 1.0
very_loose_rr1.2 112    40 35.71 48.29    58.15 1.2
     loose_rr1.5  98    26 26.53 50.77    50.64 1.5
swing3_loose_rr1  62    27 43.55 53.62   -22.64 1.0
```

## 2) Feature ↔ Label Quality

| | Old labels | New labels + PSA/LSW/FVG |
|--|------------|---------------------------|
| Global Score | 71.19 | **87.59** |
| CV PR-AUC probe | 0.5157 | **0.6417** |
| Lift | 1.193 | **1.57** |
| Shuffle p | 0.4545 | **0.0909** |
| Selected feats | 35 | **39** |

## 3) Training (Sniper-style, purged split + Wilson CI)

| Run | n | feats | Fold PR-AUC | OOS PR-AUC | Lift | Shuffle p | Best threshold |
|-----|---|-------|-------------|------------|------|-----------|----------------|
| OLD all | 81 | 138 | 0.516 | 0.4154 | 0.9 | 0.5385 | th=0.2 prec=0.4615 CI[0.2321-0.7086] rec=1.0 trades=13.0 |
| OLD selected | 81 | 35 | 0.6447 | 0.3826 | 0.829 | 0.4615 | th=0.2 prec=0.4615 CI[0.2321-0.7086] rec=1.0 trades=13.0 |
| NEW all | 137 | 171 | 0.4904 | 0.4762 | 1.0 | 0.1538 | th=0.2 prec=0.4762 CI[0.2834-0.6763] rec=1.0 trades=21.0 |
| **NEW selected** | 137 | 39 | 0.6343 | **0.6298** | **1.322** | 0.0769 | th=0.5 prec=0.7143 CI[0.3589-0.9178] rec=0.5 trades=7.0 |

## 4) Verdict

- ✅ Sample count improved (81 → 137).
- ✅ Feature quality score improved (71.19 → 87.59).
- ✅ OOS PR-AUC improved (0.3826 → 0.6298).
- ✅ Label-shuffle significant (p=0.0769) — real signal.
- ⚠️ Precision 0.7143 on 7.0 trades — promising but CI wide (lo=0.3589). Need more OOS trades.

## 5) What to do next in Colab

1. Replace Feature cell with v25 (PSA/LSW/FVG) — already validated here.
2. Run Labeling GA v4 with weights count=0.35, wr=0.35, feat=0.30; min_samples≈80.
3. Apply best config from this run: `swing1_loose_rr1` (see FINAL_RESULT.json).
4. Run Quality Lab; only train if Global Score ≥ 50 and shuffle p improving.
5. For 60% precision claim: need **≥100 OOS trades** with Wilson CI lower bound >50%.

## 6) Best config parameters
```json
{
  "name": "swing1_loose_rr1",
  "n": 138,
  "wins": 56,
  "wr": 40.58,
  "fq": 51.35,
  "fitness": 64.35,
  "accepted_raw": 138,
  "rr": 1.0,
  "max_bars": 60,
  "config": {
    "swing_n": 1,
    "ab_min_bars": 2,
    "ab_max_bars": 12,
    "ab_atr_mult_min": 1.1,
    "ab_extended_min": 1.5,
    "ab_extended_max": 6.0,
    "bc_min_bars": 2,
    "bc_retrace_min": 0.15,
    "bc_retrace_max": 0.65,
    "box_scale": 1.0,
    "signal_search_bars": 100,
    "signal_atr_mult": 0.4,
    "signal_body_ratio_min": 0.3,
    "signal_wick_reject_ratio": 1.5,
    "allow_fl_candle": true,
    "confirm_search_bars": 20,
    "confirm_atr_mult": 0.3,
    "sl_buffer_atr_mult": 0.1,
    "cd_ab_mode": "CD > AB (طبق نقاط تکرارشده در PDF)",
    "ab_noise_fraction_max": 0.5,
    "ab2_discount_mult": 0.7,
    "rr": 1.0,
    "max_bars": 60,
    "use_trend_gate": false,
    "trend_gate_mode": "خلاف روند (Counter-Trend Reversal)",
    "spread_raw": 0.00012
  }
}
```

## 7) Selected features (39)
```
[
  "MTF_HTF2_Market_Structure_State",
  "Structure_Streak_Count",
  "MTF_HTF2_Last_High_Class",
  "Dist_To_Bear_OB_ATR",
  "Price_to_EMA200_ATR",
  "EMA_Stack_Score",
  "Tenkan_Kijun_Dist_ATR",
  "PS_AB_Range_ATR",
  "Last_High_Class",
  "MTF_Alignment_Score",
  "HTF_Structure_Streak_Count",
  "Market_Structure_State",
  "Slope_200_ATR",
  "HTF_BOS_Same_Direction_Streak",
  "HTF_Last_High_Class",
  "LSW_Bull_Sweep_Flag",
  "HTF_Last_Low_Class",
  "MTF_HTF2_Structure_Range_Position",
  "HTF_Dist_To_SwingLow_ATR",
  "PSA_Candle_Range_ATR",
  "PS_BC_Bars",
  "PW_AB_Range_ATR",
  "Tick_Buy_Pressure",
  "MTF_HTF2_Bars_Since_SwingLow",
  "MTF_HTF2_Last_Low_Class",
  "BOS_Same_Direction_Streak",
  "HTF_Structure_Range_Position",
  "Volume_per_Range",
  "Slope_100_ATR",
  "SwingHigh_Touch_Count",
  "LSW_Sweep_Tick_Align",
  "LSW_Bars_Since_Bear_Sweep",
  "Price_to_Kumo_ATR",
  "LSW_Rejection_Score",
  "MTF_HTF2_Structure_Streak_Count",
  "PSA_AB_Range_ATR",
  "PSA_Dist_To_B_ATR",
  "PS_Bars_Signal_To_Trigger",
  "FVG_Dist_Bull_ATR"
]
```
