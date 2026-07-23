# @title 🎯 HIPO AI LAB [v25.0 FIXED - Safe Split + NoFillNA + WilsonCI] { display-mode: "form" }
# FIXES:
# - fillna(0) -> dropna + warmup 250
# - embargo = max_bars+10 (70) not 30
# - TimeSeriesSplit gap = embargo
# - Wilson CI in sweep
# - calibration bins min 20 samples


# @title 🎯 HIPO AI LAB [v24.0 - Threshold Sweep + Calibrated Sniper Engine] { display-mode: "form" }

# =============================================================================
# چه چیزی نسبت به v21 عوض شد و چرا:
#
# ۱. انتخاب «بهترین فولد» حذف شد.
#    نسخه‌ی قبلی از بین N فولد، مدلِ فولدی که بالاترین PR-AUC رو گرفته بود
#    نگه می‌داشت. این خودش یک نشتِ ظریف آماریه (Selection Bias روی نویز):
#    وقتی از بین چند مدل، بهترین رو روی تست انتخاب می‌کنی، عدد گزارش‌شده
#    دیگه نماینده‌ی عملکرد واقعی مدل نیست، نماینده‌ی بهترین اتفاق تصادفیه.
#    حالا: یک مدل نهایی روی کل دیتای CV (با یک برش انتهایی برای early stopping)
#    آموزش داده می‌شود؛ فولدها فقط برای گزارشِ ثبات (Stability) استفاده می‌شوند.
#
# ۲. کالیبراسیون احتمال اضافه شد.
#    scale_pos_weight برای جبران عدم‌تعادل کلاس لازمه، ولی خروجی predict_proba
#    رو غیرکالیبره (سیستماتیک بالا) می‌کنه. یعنی «آستانه‌ی اطمینان ۰.۷» دیگه
#    معنای واقعیش رو نداره. حالا بعد از آموزش، یک لایه‌ی کالیبراسیون
#    (Isotonic Regression) روی یک برش کاملاً جدا از دیتا سوار می‌شه تا
#    عددی که به‌عنوان «احتمال» می‌بینی واقعاً نزدیک به نرخ برد واقعی باشه.
#
# ۳. num_classes / class_mapping در متادیتا ذخیره می‌شه (رفع باگ v21 که باعث
#    می‌شد Post-Process Lab مدل باینری رو اشتباهی سه‌کلاسه فرض کنه).
#
# ۴. دیوار Embargo ضد نشت هنوز پابرجاست (۱۰۰ کندل فاصله بین CV و تست نهایی).
# =============================================================================

import sys, subprocess

def smart_install():
    reqs = ["gradio", "xgboost", "scikit-learn", "plotly", "pandas", "pyarrow"]
    missing = []
    for req in reqs:
        try:
            import_name = "sklearn" if req == "scikit-learn" else req
            __import__(import_name)
        except ImportError:
            missing.append(req)
    if missing:
        print(f"⏳ Installing Infrastructure ({', '.join(missing)})...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        from IPython.display import clear_output
        clear_output()

smart_install()

import os, glob, gc, traceback, shutil, json, zipfile
from datetime import datetime
import pandas as pd
import numpy as np
import gradio as gr
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
import plotly.graph_objects as go


# ====================== FIXES v25 ======================
def wilson_ci(k, n, z=1.96):
    if n==0:
        return (0.0,0.0,0.0)
    p = k/n
    denom = 1+z**2/n
    center = (p+z**2/(2*n))/denom
    half = (z*np.sqrt((p*(1-p)+z**2/(4*n))/n))/denom
    return p, max(0,center-half), min(1,center+half)

def get_safe_splits(n_total, max_bars, test_frac=0.15, calib_frac=0.15):
    embargo = max_bars + 10
    final_idx = int(n_total*(1-test_frac-calib_frac))
    calib_start = min(final_idx+embargo, n_total)
    calib_end = min(int(n_total*(1-test_frac)), n_total)
    oos_start = min(calib_end+embargo, n_total)
    return final_idx, calib_start, calib_end, oos_start, embargo

def prepare_features_safe(df_raw, drop_list):
    # FIX: no fillna(0) - drop warmup 250 + dropna
    df = df_raw.copy()
    # drop list columns first to check numeric?
    features = [c for c in df.columns if c not in drop_list and np.issubdtype(df[c].dtype, np.number)]
    # warmup
    warmup=250
    if len(df)>warmup:
        df = df.iloc[warmup:]
    # drop rows with inf
    df = df.replace([np.inf, -np.inf], np.nan)
    # drop columns with >40% NaN
    nan_frac = df[features].isna().mean()
    good_features = nan_frac[nan_frac<0.4].index.tolist()
    dropped = set(features)-set(good_features)
    if dropped:
        print(f"Dropped {len(dropped)} bad features >40% NaN: {list(dropped)[:5]}")
    df_clean = df.dropna(subset=good_features)
    return df_clean, good_features

# =======================================================

DATA_DIR = "/content/hipo_lab_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_labeled_files():
    files = glob.glob(os.path.join(DATA_DIR, "*_Labeled.parquet"))
    return [os.path.basename(f) for f in files] if files else ["No Data Found"]

def update_target_classes(dataset_names):
    if not dataset_names or "No Data" in dataset_names[0]:
        return gr.update(choices=["0", "1", "2"], value="1")
    unique_classes = set()
    try:
        for d in dataset_names:
            df_temp = pd.read_parquet(os.path.join(DATA_DIR, d), columns=['Target_Class'])
            unique_classes.update(df_temp['Target_Class'].dropna().astype(int).unique())
        choices = [str(c) for c in sorted(list(unique_classes))]
        return gr.update(choices=choices, value=choices[1] if len(choices) > 1 else choices[0])
    except Exception:
        return gr.update(choices=["0", "1", "2"], value="1")

# =============================================================================
# موتور گزارش‌دهی HTML
# =============================================================================
def generate_v24_report(model_data, feature_names, filename, focus_class):
    class_names = ["Noise / Other Classes", f"🎯 Target (Class {focus_class})"]

    fig_cm = go.Figure(data=go.Heatmap(z=model_data['cm'], x=class_names, y=class_names, colorscale='Magma', text=model_data['cm'], texttemplate="%{text}"))
    fig_cm.update_layout(title=f"Sniper Matrix (Focus: Class {focus_class} | Calibrated OOS Test)", template='plotly_dark', width=450, height=400)

    fi = model_data['fi']
    idx = np.argsort(fi)[-30:]
    fig_fi = go.Figure(go.Bar(x=fi[idx], y=[feature_names[i] for i in idx], orientation='h', marker=dict(color=fi[idx], colorscale='Electric')))
    fig_fi.update_layout(title="Top 30 Alpha Features (Sniper Weights)", template='plotly_dark', height=700)

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=model_data['calib_bins'], y=model_data['calib_bins'], mode='lines', name='کالیبراسیون کامل', line=dict(dash='dash', color='gray')))
    fig_cal.add_trace(go.Scatter(x=model_data['calib_pred'], y=model_data['calib_true'], mode='lines+markers', name='مدل ما (بعد از کالیبراسیون)', line=dict(color='#00ff88')))
    fig_cal.update_layout(title="نمودار کالیبراسیون (آیا وقتی مدل می‌گه 70٪، واقعاً 70٪ برنده‌ایم؟)",
                           xaxis_title="احتمال پیش‌بینی‌شده", yaxis_title="نرخ برد واقعی", template='plotly_dark', height=450)

    # 🎯 نمودار پیمایش آستانه: Precision و Recall و تعداد معامله در برابر Threshold
    sweep_df = model_data.get('sweep_df', pd.DataFrame())
    fig_sweep = go.Figure()
    if not sweep_df.empty:
        fig_sweep.add_trace(go.Scatter(x=sweep_df['Threshold'], y=sweep_df['Precision'], mode='lines+markers', name='Precision', line=dict(color='#00ff88')))
        fig_sweep.add_trace(go.Scatter(x=sweep_df['Threshold'], y=sweep_df['Recall'], mode='lines+markers', name='Recall', line=dict(color='#00f2ff')))
        fig_sweep.add_trace(go.Scatter(x=sweep_df['Threshold'], y=sweep_df['Trades_%'] / 100, mode='lines+markers', name='% معاملات گرفته‌شده', line=dict(color='#ffaa00', dash='dot')))
        fig_sweep.add_hline(y=0.60, line_dash="dash", line_color="red", annotation_text="هدف Precision=0.60")
    fig_sweep.update_layout(title="پیمایش آستانه: تعادل بین دقت، ریکال، و تعداد معامله",
                             xaxis_title="آستانه‌ی اطمینان (Threshold)", yaxis_title="مقدار", template='plotly_dark', height=500)

    rep_df = pd.DataFrame(model_data['report']).transpose().round(4)
    fold_str = " | ".join([f"F{i+1}: {a:.4f}" for i, a in enumerate(model_data['fold_aucs'])])
    sweep_table_html = sweep_df.to_html(classes='grid-table', index=False) if not sweep_df.empty else "<p>دیتای کافی نبود.</p>"

    html = f"""
    <!DOCTYPE html>
    <html lang="fa">
    <head>
        <meta charset="UTF-8">
        <title>HIPO SNIPER BRAIN v24.0</title>
        <style>
            body {{ background:#05080f; color:#e0e6ed; font-family:monospace; padding:30px; direction:rtl; }}
            h1 {{ color:#00ff88; text-align:center; border-bottom:1px solid #1a2433; padding-bottom:10px; text-shadow: 0 0 10px #00ff88; }}
            h3 {{ color:#00f2ff; }}
            .container {{ display:flex; flex-wrap:wrap; gap:25px; justify-content:center; margin-top:30px; }}
            .box {{ background:#0d1117; padding:20px; border-radius:15px; border:1px solid #30363d; flex:1; min-width:400px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }}
            .grid-table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
            .grid-table td, .grid-table th {{ border:1px solid #30363d; padding:8px; text-align:center; }}
            .grid-table th {{ background:#1a1f29; color:#00ff88; }}
            .footer {{ text-align:center; margin-top:30px; padding:20px; background:rgba(0,255,136,0.05); border-radius:10px; border: 1px solid #00ff88; }}
            .recommend {{ text-align:center; margin:20px auto; padding:15px; background:rgba(255,170,0,0.08); border:1px solid #ffaa00; border-radius:10px; max-width:900px; }}
        </style>
    </head>
    <body>
        <h1>🎯 HIPO SNIPER BRAIN v24.0 (Threshold Sweep + Calibrated)</h1>
        <div style="text-align:center; margin-bottom: 20px;">
            <span style="background:#1a2433; padding:10px 20px; border-radius:8px; color:#fff;">Target Focus: <b>Class {focus_class}</b> | Fold PR-AUCs: {fold_str}</span>
        </div>
        <div class="recommend"><b>💡 پیشنهاد آستانه (هدف: Precision ≥ 0.60، بیشترین Recall ممکن):</b><br>{model_data.get('recommendation','')}</div>
        <div class="container">
            <div class="box"><h3>📊 Precision & Recall (در آستانه‌ی انتخابی شما، OOS)</h3>{rep_df.to_html(classes='grid-table')}</div>
            <div class="box">{fig_cm.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        </div>
        <div class="box" style="margin-top:30px; flex:none; width:100%; box-sizing:border-box;">{fig_sweep.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div class="box" style="margin-top:30px; flex:none; width:100%; box-sizing:border-box;"><h3>📋 جدول کامل پیمایش آستانه</h3>{sweep_table_html}</div>
        <div class="box" style="margin-top:30px; flex:none; width:100%; box-sizing:border-box;">{fig_cal.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div class="box" style="margin-top:30px; flex:none; width:100%; box-sizing:border-box;">{fig_fi.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div class="footer"><p style="font-size:1.2em; color:#00ff88;">Deep PR-AUC (Precision-Recall Area, Calibrated): <b>{model_data['auc']:.5f}</b></p></div>
    </body>
    </html>
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    return filename

# =============================================================================
# موتور اصلی آموزش
# =============================================================================
def run_sniper_training(dataset_names, focus_class, n_folds, purge_gap, n_trees, stop_rounds,
                         max_depth, lr, use_gpu, user_notes, custom_threshold, progress=gr.Progress()):
    logs = []
    def log_it(msg, prog, desc):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        progress(prog, desc=desc)
        return msg

    try:
        if not dataset_names or "No Data" in dataset_names[0]:
            yield "❌ دیتاست انتخاب نشده.", "", None, None
            return

        yield log_it("📥 بارگذاری دیتا...", 0.05, "Data Sync"), "", None, None
        dfs = [pd.read_parquet(os.path.join(DATA_DIR, d)) for d in dataset_names]
        df_raw_all = pd.concat(dfs).sort_index()
        # FIXED: safe prepare
        drop_list = ['Target_Class', 'Signal_Dir', 'index', 'level_0', 'Time', 'origin_time', 'origin_price', 'entry_price', 'M1_SL', 'M1_TP', 'M2_SL', 'M2_TP', 'M3_SL', 'M3_TP', 'pair', 'max_bars', 'Raw_ATR', 'rev_cross_idx', 'B_SL', 'B_TP', 'S_SL', 'S_TP', 'Open', 'High', 'Low', 'Close', 'Bid']
        df, features = prepare_features_safe(df_raw_all, drop_list)
        # if prepare returned features, use it, else compute
        if len(features)==0:
            features = [c for c in df.columns if c not in drop_list and np.issubdtype(df[c].dtype, np.number)]

        X = df[features].astype('float32')

        target_c = int(focus_class)
        y = (df['Target_Class'].astype(int) == target_c).astype(int)

        # 🛑 دیوار ضد نشت: سه بخش کاملاً مجزا با فاصله‌ی Embargo واقعی بین‌شون
        #
        # ⚠️ باگ نسخه‌ی v22 اینجا بود: از همون متغیر embargo_bars (فقط ۱۰۰ ردیف)
        # هم برای «فاصله‌ی امنیتی» و هم به‌اشتباه برای «اندازه‌ی کل برش کالیبراسیون»
        # استفاده می‌شد. یعنی کالیبراسیون فقط روی ~۱۰۰ ردیف (که با نرخ برد ۲۹٪
        # شاید ۲۵-۳۰ نمونه‌ی مثبت باشه) انجام می‌شد — برای بین‌های احتمال بالا
        # (مثلاً حوالی ۰.۷۵) شاید فقط ۶-۱۰ نمونه وجود داشت، که هیچ اعتباری نداره.
        # دقیقاً همون چیزی که باعث می‌شد نمودار کالیبراسیون در بخش «اطمینان بالا»
        # (0.75 -> فقط 12.5% برد واقعی) کاملاً گمراه‌کننده باشد.
        #
        # حالا: embargo_gap فقط یک فاصله‌ی امنیتی کوچیکه (جلوگیری از نشت فیچرهای
        # rolling)، و کالیبراسیون یک برش واقعی و معنادار (٪۱۵ از کل دیتا) می‌گیرد.
        # FIXED: embargo = max_bars + 10, not 30, to avoid leakage
        max_bars_for_embargo = 70  # conservative, should be param but fixed here; ideally = max_bars from labeling
        n_total = len(df)
        final_idx, calib_start_idx, calib_end_idx, oos_start_idx, embargo_gap = get_safe_splits(n_total, max_bars_for_embargo)
        print(f"Safe splits: total={n_total} final_idx={final_idx} calib={calib_start_idx}:{calib_end_idx} oos={oos_start_idx}: embargo={embargo_gap}")

        X_cv, y_cv = X.iloc[:final_idx], y.iloc[:final_idx]
        X_calib = np.ascontiguousarray(X.iloc[calib_start_idx:calib_end_idx], dtype=np.float32)
        y_calib = y.iloc[calib_start_idx:calib_end_idx].values.astype(int)
        X_final_test = np.ascontiguousarray(X.iloc[oos_start_idx:], dtype=np.float32)
        y_final_test = y.iloc[oos_start_idx:].values.astype(int)

        n_calib_pos = int(y_calib.sum())
        yield log_it(f"⚖️ اندازه‌ی برش کالیبراسیون: {len(y_calib)} ردیف ({n_calib_pos} مثبت). اندازه‌ی تست نهایی: {len(y_final_test)} ردیف.", 0.11, "Split Sizing"), "\n".join(logs), None, None
        if n_calib_pos < 30:
            yield log_it(f"⚠️ هشدار: فقط {n_calib_pos} نمونه‌ی مثبت در برش کالیبراسیون — نمودار کالیبراسیون (به‌خصوص بین‌های احتمال بالا) ممکن است هنوز پرنویز باشد.", 0.12, "Low Sample Warning"), "\n".join(logs), None, None

        total_target, total_noise = y_cv.sum(), len(y_cv) - y_cv.sum()
        yield log_it(f"🔍 در منطقه‌ی CV: {total_target} Target در برابر {total_noise} Noise.", 0.09, "Target Analysis"), "\n".join(logs), None, None
        if total_target == 0:
            yield log_it(f"❌ هیچ نمونه‌ای از کلاس {target_c} وجود ندارد!", 0.99, "Error"), "\n".join(logs), None, None
            return

        device_target = 'cuda' if use_gpu else 'cpu'
        base_params = dict(max_depth=int(max_depth), learning_rate=float(lr), tree_method='hist',
                            device=device_target, booster='gbtree', objective='binary:logistic',
                            eval_metric='aucpr', random_state=42, n_jobs=-1)

        # --- فاز ۱: فولدهای CV فقط برای سنجش ثبات (Stability)، نه انتخاب مدل ---
        # FIXED: gap = embargo to avoid overlap of label horizons
        tscv = TimeSeriesSplit(n_splits=int(n_folds), gap=embargo_gap)
        fold_aucs = []
        yield log_it(f"🛡️ سنجش ثبات مدل در {n_folds} فولد...", 0.1, "Stability Check"), "\n".join(logs), None, None

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_cv), 1):
            fold_prog = 0.1 + (fold / n_folds) * 0.35
            X_tr = np.ascontiguousarray(X_cv.iloc[train_idx], dtype=np.float32)
            y_tr = y_cv.iloc[train_idx].values.astype(int)
            X_te = np.ascontiguousarray(X_cv.iloc[test_idx], dtype=np.float32)
            y_te = y_cv.iloc[test_idx].values.astype(int)

            spw = (y_tr == 0).sum() / ((y_tr == 1).sum() + 1e-9)
            fold_model = xgb.XGBClassifier(**base_params, n_estimators=int(n_trees), scale_pos_weight=spw,
                                            early_stopping_rounds=int(stop_rounds))
            fold_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            auc = average_precision_score(y_te, fold_model.predict_proba(X_te)[:, 1])
            fold_aucs.append(auc)
            yield log_it(f"✅ فولد {fold} | PR-AUC: {auc:.4f}", fold_prog, f"Fold {fold}/{n_folds}"), "\n".join(logs), None, None
            gc.collect()

        auc_std = float(np.std(fold_aucs))
        auc_mean = float(np.mean(fold_aucs))
        yield log_it(f"📐 میانگین PR-AUC فولدها: {auc_mean:.4f} ± {auc_std:.4f} (پراکندگی بالا = بی‌ثباتی)", 0.46, "Stability Summary"), "\n".join(logs), None, None

        # --- فاز ۲: یک مدل نهایی روی کل CV، بدون Cherry-Picking ---
        yield log_it("🎯 آموزش مدل نهایی روی کل دیتای CV...", 0.5, "Final Fit"), "\n".join(logs), None, None
        val_cut = int(len(X_cv) * 0.9)
        X_fit, y_fit = X_cv.iloc[:val_cut], y_cv.iloc[:val_cut]
        X_val, y_val = X_cv.iloc[val_cut:], y_cv.iloc[val_cut:]
        spw_final = (y_fit == 0).sum() / ((y_fit == 1).sum() + 1e-9)

        final_model = xgb.XGBClassifier(**base_params, n_estimators=int(n_trees), scale_pos_weight=spw_final,
                                         early_stopping_rounds=int(stop_rounds))
        final_model.fit(np.ascontiguousarray(X_fit, dtype=np.float32), y_fit.values.astype(int),
                         eval_set=[(np.ascontiguousarray(X_val, dtype=np.float32), y_val.values.astype(int))],
                         verbose=False)

        # --- فاز ۳: کالیبراسیون احتمال روی یک برش کاملاً مجزا ---
        yield log_it("⚖️ کالیبراسیون احتمالات (Isotonic)...", 0.7, "Calibration"), "\n".join(logs), None, None
        raw_calib_prob = final_model.predict_proba(X_calib)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        if len(np.unique(y_calib)) > 1:
            calibrator.fit(raw_calib_prob, y_calib)
        else:
            yield log_it("⚠️ برش کالیبراسیون فقط یک کلاس داشت؛ کالیبراسیون رد شد (نتایج خام باقی می‌مانند).", 0.72, "Calibration Skipped"), "\n".join(logs), None, None
            calibrator = None

        # --- فاز ۴: آزمون نهایی روی دیتای کاملاً قرنطینه‌شده ---
        yield log_it("🏁 آزمون نهایی OOS...", 0.85, "Final OOS Test"), "\n".join(logs), None, None
        raw_prob_final = final_model.predict_proba(X_final_test)[:, 1]
        calib_prob_final = calibrator.predict(raw_prob_final) if calibrator is not None else raw_prob_final

        y_pred = (calib_prob_final >= float(custom_threshold)).astype(int)
        final_oos_auc = average_precision_score(y_final_test, calib_prob_final)

        # --- 🎯 پیمایش آستانه (Threshold Sweep) ---
        # چرا این لازم بود: بعد از رفع باگ کالیبراسیون، مدل در آستانه‌ی ۰.۵ خیلی
        # کم سیگنال می‌داد. این خودش لزوماً باگ نیست — یک مدل درست‌کالیبره‌شده با
        # نرخ برد پایه‌ی ~۲۹٪ طبیعتاً به‌ندرت احتمال بالای ۵۰٪ می‌ده. اما به‌جای
        # حدس‌زدن و آموزش دوباره برای هر آستانه، این‌جا یک‌بار همه‌ی آستانه‌ها را
        # روی همین احتمال‌های کالیبره‌شده‌ی از‌قبل‌محاسبه‌شده می‌سنجیم.
        yield log_it("📐 پیمایش آستانه‌ی اطمینان (Threshold Sweep)...", 0.9, "Threshold Sweep"), "\n".join(logs), None, None
        sweep_rows = []
        for th in np.arange(0.15, 0.86, 0.025):
            pred_th = (calib_prob_final >= th).astype(int)
            trades = int(pred_th.sum())
            if trades == 0:
                continue
            tp = int(((pred_th==1) & (y_final_test==1)).sum())
            prec_p, prec_lo, prec_hi = wilson_ci(tp, trades)
            prec = precision_score(y_final_test, pred_th, zero_division=0)
            rec = recall_score(y_final_test, pred_th, zero_division=0)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            sweep_rows.append({
                "Threshold": round(float(th), 3), "Precision": round(float(prec), 4),
                "Precision_CI_Lo": round(float(prec_lo),4), "Precision_CI_Hi": round(float(prec_hi),4),
                "Recall": round(float(rec), 4), "Trades": trades,
                "Trades_%": round(100 * trades / len(y_final_test), 2), "F1": round(float(f1), 4)
            })
        sweep_df = pd.DataFrame(sweep_rows)

        # پیشنهاد خودکار: هدف کاربر Precision >= 0.60 با بیشترین Recall ممکن
        target_precision = 0.60
        recommendation_txt = "دیتای کافی برای پیشنهاد آستانه وجود نداشت."
        if not sweep_df.empty:
            candidates = sweep_df[sweep_df["Precision"] >= target_precision]
            if not candidates.empty:
                best_row = candidates.loc[candidates["Recall"].idxmax()]
                recommendation_txt = (f"✅ در آستانه‌ی {best_row['Threshold']}: Precision={best_row['Precision']} | "
                                       f"Recall={best_row['Recall']} | تعداد معامله={best_row['Trades']} از {len(y_final_test)} "
                                       f"({best_row['Trades_%']}٪ سیگنال‌ها)")
            else:
                best_row = sweep_df.loc[sweep_df["Precision"].idxmax()]
                recommendation_txt = (f"⚠️ در این دیتا هیچ آستانه‌ای به Precision {target_precision} نرسید. "
                                       f"بهترین Precision قابل‌دسترس: {best_row['Precision']} در آستانه‌ی {best_row['Threshold']} "
                                       f"(Recall={best_row['Recall']}, تعداد معامله={best_row['Trades']}). "
                                       f"یعنی لبه‌ی مدل برای این هدف کافی نیست؛ باید فیچر/لیبل تقویت بشه، نه فقط آستانه عوض بشه.")
        yield log_it(recommendation_txt, 0.93, "Threshold Recommendation"), "\n".join(logs), None, None

        # منحنی کالیبراسیون برای گزارش (تقسیم به ۱۰ بین مساوی)
        # FIXED: بین‌بندی کوانتایلی + حداقل 20 نمونه
        n_bins = min(8, max(2, len(calib_prob_final) // 20))
        try:
            bin_edges = np.unique(np.quantile(calib_prob_final, np.linspace(0, 1, n_bins + 1)))
        except Exception:
            bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.digitize(calib_prob_final, bin_edges[1:-1])
        calib_pred_pts, calib_true_pts = [], []
        for b in range(len(bin_edges)):
            mask = bin_idx == b
            if mask.sum() >= 20:
                calib_pred_pts.append(calib_prob_final[mask].mean())
                calib_true_pts.append(y_final_test[mask].mean())

        model_data = {
            'cm': confusion_matrix(y_final_test, y_pred, labels=[0, 1]),
            'report': classification_report(y_final_test, y_pred, labels=[0, 1],
                                             target_names=["Noise/Other", f"🎯 Target (Class {target_c})"],
                                             output_dict=True, zero_division=0),
            'fi': final_model.feature_importances_,
            'auc': final_oos_auc,
            'fold_aucs': fold_aucs,
            'calib_pred': calib_pred_pts, 'calib_true': calib_true_pts, 'calib_bins': [0, 1],
            'sweep_df': sweep_df, 'recommendation': recommendation_txt
        }

        report_path = os.path.join(DATA_DIR, "Sniper_Report_v24.html")
        generate_v24_report(model_data, features, report_path, focus_class)

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        master_model_temp = os.path.join(DATA_DIR, "master_model.json")
        final_model.save_model(master_model_temp)

        calib_temp = os.path.join(DATA_DIR, "calibrator.json")
        if calibrator is not None:
            with open(calib_temp, 'w', encoding='utf-8') as f:
                json.dump({"x": calibrator.X_thresholds_.tolist(), "y": calibrator.y_thresholds_.tolist()}, f)

        meta_data = {
            "training_date": timestamp_str, "user_notes": user_notes, "features": features,
            "focused_class": int(focus_class), "num_classes": 2,
            "class_mapping": {"0": "Noise/Other", "1": f"Target (Class {focus_class})"},
            "n_folds": int(n_folds), "purge_gap": int(purge_gap), "n_trees": int(n_trees),
            "stop_rounds": int(stop_rounds), "max_depth": int(max_depth), "lr": float(lr),
            "custom_threshold": float(custom_threshold), "fold_pr_aucs": fold_aucs,
            "fold_auc_mean": auc_mean, "fold_auc_std": auc_std,
            "final_oos_pr_auc_calibrated": float(final_oos_auc), "is_calibrated": calibrator is not None
        }
        meta_temp = os.path.join(DATA_DIR, "model_metadata.json")
        with open(meta_temp, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)

        zip_filename = f"HIPO_SniperBrain_{timestamp_str}.zip"
        zip_filepath = os.path.join(DATA_DIR, zip_filename)
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(master_model_temp, "master_model.json")
            zf.write(meta_temp, "model_metadata.json")
            if calibrator is not None:
                zf.write(calib_temp, "calibrator.json")

        final_msg = (f"### 💎 آموزش تمام شد (بدون Cherry-Picking، با کالیبراسیون + پیمایش آستانه)\n"
                     f"**PR-AUC فولدها:** {auc_mean:.4f} ± {auc_std:.4f} (پراکندگی بالا یعنی بی‌ثبات)\n"
                     f"**PR-AUC نهایی روی دیتای قرنطینه‌شده (کالیبره‌شده):** {final_oos_auc:.5f}\n"
                     f"**تعداد فیچرها:** {len(features)}\n\n"
                     f"**💡 پیشنهاد آستانه:** {recommendation_txt}\n\n"
                     f"📋 جدول کامل همه‌ی آستانه‌ها (Precision/Recall/تعداد معامله) داخل گزارش HTML هست — "
                     f"دیگه لازم نیست برای هر آستانه دوباره آموزش بدی.")
        yield final_msg, "\n".join(logs), report_path, zip_filepath

    except Exception:
        err_msg = traceback.format_exc()
        yield "❌ کرش در موتور آموزش (لاگ زیر را ببین)", err_msg, None, None

# =============================================================================
# رابط کاربری
# =============================================================================
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.HTML("<div style='text-align:center; padding:20px; background:#05080f; border-radius:15px; border:1px solid #1a2433;'>" +
            "<h1 style='color:#00ff88; margin:0; font-family:monospace;'>🎯 HIPO AI: SNIPER TRAINING v24.0</h1>" +
            "<p style='color:#8b949e;'>No Cherry-Picking | Isotonic Calibration | Zero-Leakage Embargo</p></div>")

    with gr.Row():
        with gr.Column(scale=2):
            w_dataset = gr.CheckboxGroup(choices=get_labeled_files(), label="📁 دیتاست‌های لیبل‌گذاری‌شده", value=[get_labeled_files()[0]] if get_labeled_files() else [])
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 تنظیمات هدف")
            w_focus = gr.Dropdown(choices=["0", "1", "2"], label="کلاس هدف (بقیه = نویز)", value="1")
            w_threshold = gr.Slider(minimum=0.3, maximum=0.9, step=0.01, value=0.5, label="آستانه‌ی اطمینان اولیه (بعداً در Post-Process قابل تغییره)")

    with gr.Row():
        w_folds = gr.Slider(minimum=3, maximum=10, step=1, value=5, label="تعداد فولد (فقط برای سنجش ثبات)")
        w_purge = gr.Slider(minimum=0, maximum=200, step=10, value=50, label="Purge Gap")
        w_trees = gr.Slider(minimum=100, maximum=3000, step=100, value=1000, label="تعداد درخت‌ها")
        w_stop = gr.Slider(minimum=10, maximum=200, step=10, value=50, label="Early Stopping Rounds")
        w_depth = gr.Slider(minimum=2, maximum=10, step=1, value=5, label="Max Depth")
        w_lr = gr.Slider(minimum=0.005, maximum=0.3, step=0.005, value=0.03, label="Learning Rate")
        w_gpu = gr.Checkbox(label="استفاده از GPU", value=True)
        w_notes = gr.Textbox(label="یادداشت آزمایش", value="")

    w_btn = gr.Button("🔥 START CALIBRATED SNIPER TRAINING", variant="primary", size="lg")
    w_msg = gr.Markdown()
    w_log = gr.Textbox(label="لاگ کامل", lines=10)
    w_report = gr.File(label="📄 گزارش HTML")
    w_zip = gr.File(label="📦 مدل + متادیتا + کالیبراتور")

    w_dataset.change(update_target_classes, inputs=[w_dataset], outputs=[w_focus])
    w_btn.click(run_sniper_training,
                inputs=[w_dataset, w_focus, w_folds, w_purge, w_trees, w_stop, w_depth, w_lr, w_gpu, w_notes, w_threshold],
                outputs=[w_msg, w_log, w_report, w_zip])

app.queue().launch(share=True, inbrowser=True)