
# @title 🎯 HIPO AI LAB [v26.0 FINAL - Sniper + Doctor + WilsonCI + Safe Split] { display-mode: "form" }
# =============================================================================
# نسخه نهایی کامل - شامل:
# 1. آموزش Sniper FIXED (بدون fillna(0), embargo=70, Wilson CI)
# 2. دکتر HIPO DOCTOR یکپارچه (تب دوم) - تمام 8 بخش تشخیص + گزارش HTML
# 3. Threshold Sweep با CI
# 4. گزارش HTML کامل با نمودارهای Calibration + Sweep + Feature Importance
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

import os, glob, gc, traceback, shutil, json, zipfile, warnings, io
from datetime import datetime
import pandas as pd
import numpy as np
import gradio as gr
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

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

# ---------- FIXED Helpers ----------
def wilson_ci(k, n, z=1.96):
    if n==0:
        return (0.0,0.0,0.0)
    p = k/n
    denom = 1+z**2/n
    center = (p+z**2/(2*n))/denom
    half = (z*np.sqrt((p*(1-p)+z**2/(4*n))/n))/denom
    return p, max(0,center-half), min(1,center+half)

def two_prop_ztest(k1,n1,k2,n2):
    if n1==0 or n2==0:
        return np.nan, np.nan
    p1,p2 = k1/n1, k2/n2
    p_pool = (k1+k2)/(n1+n2)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    if se==0:
        return 0.0, 1.0
    z = (p1-p2)/se
    from math import erf, sqrt
    p_val = 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
    return z, p_val

def get_safe_splits(n_total, max_bars, test_frac=0.15, calib_frac=0.15):
    embargo = max_bars+10
    final_idx = int(n_total*(1-test_frac-calib_frac))
    calib_start = min(final_idx+embargo, n_total)
    calib_end = min(int(n_total*(1-test_frac)), n_total)
    oos_start = min(calib_end+embargo, n_total)
    return final_idx, calib_start, calib_end, oos_start, embargo

def prepare_features_safe(df_raw, drop_list):
    features = [c for c in df_raw.columns if c not in drop_list and np.issubdtype(df_raw[c].dtype, np.number)]
    warmup=250
    if len(df_raw)>warmup:
        df_raw = df_raw.iloc[warmup:]
    df_raw = df_raw.replace([np.inf, -np.inf], np.nan)
    nan_frac = df_raw[features].isna().mean()
    good_features = nan_frac[nan_frac<0.4].index.tolist()
    dropped = set(features)-set(good_features)
    if dropped:
        print(f"Dropped {len(dropped)} bad features >40% NaN: {list(dropped)[:5]}")
    df_clean = df_raw.dropna(subset=good_features)
    return df_clean, good_features

# ---------- HTML Report ----------
def generate_v26_report(model_data, feature_names, filename, focus_class):
    class_names = ["Noise / Other Classes", f"Target (Class {focus_class})"]
    fig_cm = go.Figure(data=go.Heatmap(z=model_data['cm'], x=class_names, y=class_names, colorscale='Magma', text=model_data['cm'], texttemplate="%{text}"))
    fig_cm.update_layout(title=f"Sniper Matrix (Focus: Class {focus_class} | Calibrated OOS)", template='plotly_dark', width=450, height=400)
    fi = model_data['fi']
    idx = np.argsort(fi)[-30:]
    fig_fi = go.Figure(go.Bar(x=fi[idx], y=[feature_names[i] for i in idx], orientation='h', marker=dict(color=fi[idx], colorscale='Electric')))
    fig_fi.update_layout(title="Top 30 Alpha Features", template='plotly_dark', height=700)
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Perfect Calib', line=dict(dash='dash', color='gray')))
    fig_cal.add_trace(go.Scatter(x=model_data['calib_pred'], y=model_data['calib_true'], mode='lines+markers', name='Model (calibrated)', line=dict(color='#00ff88')))
    fig_cal.update_layout(title="Calibration (آیا 70% واقعا 70% می‌برد؟)", xaxis_title="احتمال پیش‌بینی", yaxis_title="نرخ برد واقعی", template='plotly_dark', height=450)
    sweep_df = model_data.get('sweep_df', pd.DataFrame())
    fig_sweep = go.Figure()
    if not sweep_df.empty:
        fig_sweep.add_trace(go.Scatter(x=sweep_df['Threshold'], y=sweep_df['Precision'], mode='lines+markers', name='Precision', line=dict(color='#00ff88')))
        fig_sweep.add_trace(go.Scatter(x=sweep_df['Threshold'], y=sweep_df['Recall'], mode='lines+markers', name='Recall', line=dict(color='#00f2ff')))
        fig_sweep.add_trace(go.Scatter(x=sweep_df['Threshold'], y=sweep_df['Trades_%']/100, mode='lines+markers', name='% Trades', line=dict(color='#ffaa00', dash='dot')))
        fig_sweep.add_hline(y=0.60, line_dash="dash", line_color="red", annotation_text="هدف 60%")
    fig_sweep.update_layout(title="Threshold Sweep: دقت vs ریکال vs تعداد معامله (با Wilson CI)", xaxis_title="Threshold", yaxis_title="مقدار", template='plotly_dark', height=500)
    rep_df = pd.DataFrame(model_data['report']).transpose().round(4)
    fold_str = " | ".join([f"F{i+1}: {a:.4f}" for i,a in enumerate(model_data['fold_aucs'])])
    sweep_table_html = sweep_df.to_html(classes='grid-table', index=False) if not sweep_df.empty else "<p>دیتای کافی نبود.</p>"
    html = f"""
    <!DOCTYPE html><html lang="fa"><head><meta charset="UTF-8"><title>HIPO SNIPER v26 FINAL</title>
    <style>body{{background:#05080f;color:#e0e6ed;font-family:monospace;padding:30px;direction:rtl;}}h1{{color:#00ff88;text-align:center;border-bottom:1px solid #1a2433;padding-bottom:10px;}}h3{{color:#00f2ff;}}.container{{display:flex;flex-wrap:wrap;gap:25px;justify-content:center;margin-top:30px;}}.box{{background:#0d1117;padding:20px;border-radius:15px;border:1px solid #30363d;flex:1;min-width:400px;}}.grid-table{{width:100%;border-collapse:collapse;font-size:0.85em;}}.grid-table td,.grid-table th{{border:1px solid #30363d;padding:8px;text-align:center;}}.grid-table th{{background:#1a1f29;color:#00ff88;}}.footer{{text-align:center;margin-top:30px;padding:20px;background:rgba(0,255,136,0.05);border-radius:10px;border:1px solid #00ff88;}}.recommend{{text-align:center;margin:20px auto;padding:15px;background:rgba(255,170,0,0.08);border:1px solid #ffaa00;border-radius:10px;max-width:900px;}}</style></head>
    <body><h1>HIPO SNIPER BRAIN v26 FINAL (Sniper + Doctor)</h1>
    <div style="text-align:center;"><span style="background:#1a2433;padding:10px 20px;border-radius:8px;">Target: Class {focus_class} | Fold PR-AUCs: {fold_str}</span></div>
    <div class="recommend"><b>پیشنهاد آستانه (Wilson CI):</b><br>{model_data.get('recommendation','')}</div>
    <div class="container"><div class="box"><h3>Precision & Recall (OOS)</h3>{rep_df.to_html(classes='grid-table')}</div><div class="box">{fig_cm.to_html(full_html=False, include_plotlyjs='cdn')}</div></div>
    <div class="box" style="margin-top:30px;width:100%;">{fig_sweep.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="box" style="margin-top:30px;width:100%;"><h3>جدول کامل Threshold با CI</h3>{sweep_table_html}</div>
    <div class="box" style="margin-top:30px;width:100%;">{fig_cal.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="box" style="margin-top:30px;width:100%;">{fig_fi.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    <div class="footer"><p style="font-size:1.2em;color:#00ff88;">Deep PR-AUC Calibrated: <b>{model_data['auc']:.5f}</b></p></div>
    </body></html>
    """
    with open(filename,'w',encoding='utf-8') as f:
        f.write(html)
    return filename

def run_sniper_training(dataset_names, focus_class, n_folds, purge_gap, n_trees, stop_rounds, max_depth, lr, use_gpu, user_notes, custom_threshold, progress=gr.Progress()):
    logs=[]
    def log_it(msg,prog,desc):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        progress(prog,desc=desc)
        return msg
    try:
        if not dataset_names or "No Data" in dataset_names[0]:
            yield "❌ دیتاست انتخاب نشده.", "", None, None
            return
        yield log_it("📥 بارگذاری دیتا...",0.05,"Data Sync"), "", None, None
        dfs = [pd.read_parquet(os.path.join(DATA_DIR, d)) for d in dataset_names]
        df_raw_all = pd.concat(dfs).sort_index()
        drop_list = ['Target_Class', 'Signal_Dir', 'index', 'level_0', 'Time', 'origin_time', 'origin_price', 'entry_price', 'M1_SL', 'M1_TP', 'M2_SL', 'M2_TP', 'M3_SL', 'M3_TP', 'pair', 'max_bars', 'Raw_ATR', 'rev_cross_idx', 'B_SL', 'B_TP', 'S_SL', 'S_TP', 'Open', 'High', 'Low', 'Close', 'Bid']
        df, features = prepare_features_safe(df_raw_all, drop_list)
        if len(features)==0:
            features = [c for c in df.columns if c not in drop_list and np.issubdtype(df[c].dtype, np.number)]
        X = df[features].astype('float32')
        target_c = int(focus_class)
        y = (df['Target_Class'].astype(int)==target_c).astype(int)

        max_bars_for_embargo = 70
        n_total = len(df)
        final_idx, calib_start_idx, calib_end_idx, oos_start_idx, embargo_gap = get_safe_splits(n_total, max_bars_for_embargo)
        print(f"Safe splits: total={n_total} final={final_idx} calib={calib_start_idx}:{calib_end_idx} oos={oos_start_idx} embargo={embargo_gap}")

        X_cv, y_cv = X.iloc[:final_idx], y.iloc[:final_idx]
        X_calib = np.ascontiguousarray(X.iloc[calib_start_idx:calib_end_idx], dtype=np.float32)
        y_calib = y.iloc[calib_start_idx:calib_end_idx].values.astype(int)
        X_final_test = np.ascontiguousarray(X.iloc[oos_start_idx:], dtype=np.float32)
        y_final_test = y.iloc[oos_start_idx:].values.astype(int)

        n_calib_pos = int(y_calib.sum())
        yield log_it(f"⚖️ کالیراسیون: {len(y_calib)} ردیف ({n_calib_pos} مثبت). تست: {len(y_final_test)}.",0.11,"Split Sizing"), "\n".join(logs), None, None
        if n_calib_pos<30:
            yield log_it(f"⚠️ فقط {n_calib_pos} مثبت در کالیراسیون — CI پرنویز می‌شود.",0.12,"Warning"), "\n".join(logs), None, None

        total_target, total_noise = y_cv.sum(), len(y_cv)-y_cv.sum()
        yield log_it(f"🔍 CV: {total_target} Target vs {total_noise} Noise.",0.09,"Target"), "\n".join(logs), None, None
        if total_target==0:
            yield log_it(f"❌ کلاس {target_c} وجود ندارد!",0.99,"Error"), "\n".join(logs), None, None
            return

        device_target = 'cuda' if use_gpu else 'cpu'
        base_params = dict(max_depth=int(max_depth), learning_rate=float(lr), tree_method='hist', device=device_target, booster='gbtree', objective='binary:logistic', eval_metric='aucpr', random_state=42, n_jobs=-1)

        tscv = TimeSeriesSplit(n_splits=int(n_folds), gap=embargo_gap)
        fold_aucs=[]
        yield log_it(f"🛡️ سنجش ثبات در {n_folds} فولد با gap={embargo_gap}...",0.1,"Stability"), "\n".join(logs), None, None
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_cv),1):
            X_tr = np.ascontiguousarray(X_cv.iloc[train_idx], dtype=np.float32)
            y_tr = y_cv.iloc[train_idx].values.astype(int)
            X_te = np.ascontiguousarray(X_cv.iloc[test_idx], dtype=np.float32)
            y_te = y_cv.iloc[test_idx].values.astype(int)
            spw = (y_tr==0).sum()/((y_tr==1).sum()+1e-9)
            fold_model = xgb.XGBClassifier(**base_params, n_estimators=int(n_trees), scale_pos_weight=spw, early_stopping_rounds=int(stop_rounds))
            fold_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            auc = average_precision_score(y_te, fold_model.predict_proba(X_te)[:,1])
            fold_aucs.append(auc)
            yield log_it(f"✅ فولد {fold} | PR-AUC: {auc:.4f}", 0.1+(fold/n_folds)*0.35, f"Fold {fold}"), "\n".join(logs), None, None
            gc.collect()

        auc_mean=float(np.mean(fold_aucs)); auc_std=float(np.std(fold_aucs))
        yield log_it(f"📐 میانگین PR-AUC: {auc_mean:.4f} ± {auc_std:.4f}",0.46,"Stability"), "\n".join(logs), None, None

        yield log_it("🎯 آموزش مدل نهایی روی کل CV...",0.5,"Final Fit"), "\n".join(logs), None, None
        val_cut=int(len(X_cv)*0.9)
        X_fit, y_fit = X_cv.iloc[:val_cut], y_cv.iloc[:val_cut]
        X_val, y_val = X_cv.iloc[val_cut:], y_cv.iloc[val_cut:]
        spw_final=(y_fit==0).sum()/((y_fit==1).sum()+1e-9)
        final_model=xgb.XGBClassifier(**base_params, n_estimators=int(n_trees), scale_pos_weight=spw_final, early_stopping_rounds=int(stop_rounds))
        final_model.fit(np.ascontiguousarray(X_fit,dtype=np.float32), y_fit.values.astype(int), eval_set=[(np.ascontiguousarray(X_val,dtype=np.float32), y_val.values.astype(int))], verbose=False)

        yield log_it("⚖️ کالیبراسیون Isotonic...",0.7,"Calibration"), "\n".join(logs), None, None
        raw_calib_prob=final_model.predict_proba(X_calib)[:,1]
        calibrator=IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        if len(np.unique(y_calib))>1:
            calibrator.fit(raw_calib_prob, y_calib)
        else:
            calibrator=None
            yield log_it("⚠️ کالیراسیون رد شد (یک کلاس).",0.72,"Skip"), "\n".join(logs), None, None

        yield log_it("🏁 آزمون نهایی OOS...",0.85,"OOS"), "\n".join(logs), None, None
        raw_prob_final=final_model.predict_proba(X_final_test)[:,1]
        calib_prob_final=calibrator.predict(raw_prob_final) if calibrator is not None else raw_prob_final
        y_pred=(calib_prob_final>=float(custom_threshold)).astype(int)
        final_oos_auc=average_precision_score(y_final_test, calib_prob_final)

        yield log_it("📐 پیمایش آستانه با Wilson CI...",0.9,"Sweep"), "\n".join(logs), None, None
        sweep_rows=[]
        for th in np.arange(0.15,0.86,0.025):
            pred_th=(calib_prob_final>=th).astype(int)
            trades=int(pred_th.sum())
            if trades==0:
                continue
            tp=int(((pred_th==1)&(y_final_test==1)).sum())
            p_ci, lo, hi = wilson_ci(tp, trades)
            prec=precision_score(y_final_test, pred_th, zero_division=0)
            rec=recall_score(y_final_test, pred_th, zero_division=0)
            f1=(2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
            sweep_rows.append({"Threshold":round(float(th),3), "Precision":round(float(prec),4), "Precision_CI_Lo":round(lo,4), "Precision_CI_Hi":round(hi,4), "Recall":round(float(rec),4), "Trades":trades, "Trades_%":round(100*trades/len(y_final_test),2), "F1":round(float(f1),4)})
        sweep_df=pd.DataFrame(sweep_rows)

        target_precision=0.60
        recommendation_txt="دیتای کافی نبود."
        if not sweep_df.empty:
            candidates=sweep_df[sweep_df["Precision"]>=target_precision]
            if not candidates.empty:
                best_row=candidates.loc[candidates["Recall"].idxmax()]
                recommendation_txt=f"✅ در آستانه {best_row['Threshold']}: Precision={best_row['Precision']} (CI {best_row['Precision_CI_Lo']}-{best_row['Precision_CI_Hi']}) | Recall={best_row['Recall']} | Trades={best_row['Trades']} ({best_row['Trades_%']}%)"
            else:
                best_row=sweep_df.loc[sweep_df["Precision"].idxmax()]
                recommendation_txt=f"⚠️ هیچ آستانه‌ای به {target_precision} نرسید. بهترین: {best_row['Precision']} (CI {best_row['Precision_CI_Lo']}-{best_row['Precision_CI_Hi']}) در {best_row['Threshold']} (Recall={best_row['Recall']}, Trades={best_row['Trades']}). لبه کافی نیست، فیچر/لیبل تقویت شود."

        yield log_it(recommendation_txt,0.93,"Recommendation"), "\n".join(logs), None, None

        n_bins=min(8,max(2,len(calib_prob_final)//20))
        try:
            bin_edges=np.unique(np.quantile(calib_prob_final, np.linspace(0,1,n_bins+1)))
        except Exception:
            bin_edges=np.linspace(0,1,n_bins+1)
        bin_idx=np.digitize(calib_prob_final, bin_edges[1:-1])
        calib_pred_pts, calib_true_pts=[],[]
        for b in range(len(bin_edges)):
            mask=bin_idx==b
            if mask.sum()>=20:
                calib_pred_pts.append(calib_prob_final[mask].mean())
                calib_true_pts.append(y_final_test[mask].mean())

        model_data={'cm':confusion_matrix(y_final_test,y_pred,labels=[0,1]),'report':classification_report(y_final_test,y_pred,labels=[0,1],target_names=["Noise/Other",f"Target (Class {target_c})"],output_dict=True,zero_division=0),'fi':final_model.feature_importances_,'auc':final_oos_auc,'fold_aucs':fold_aucs,'calib_pred':calib_pred_pts,'calib_true':calib_true_pts,'calib_bins':[0,1],'sweep_df':sweep_df,'recommendation':recommendation_txt}

        report_path=os.path.join(DATA_DIR,"Sniper_Report_v26_FINAL.html")
        generate_v26_report(model_data, features, report_path, focus_class)

        timestamp_str=datetime.now().strftime('%Y%m%d_%H%M%S')
        master_model_temp=os.path.join(DATA_DIR,"master_model.json")
        final_model.save_model(master_model_temp)
        calib_temp=os.path.join(DATA_DIR,"calibrator.json")
        if calibrator is not None:
            with open(calib_temp,'w',encoding='utf-8') as f:
                json.dump({"x":calibrator.X_thresholds_.tolist(),"y":calibrator.y_thresholds_.tolist()}, f)

        meta_data={"training_date":timestamp_str,"user_notes":user_notes,"features":features,"focused_class":int(focus_class),"num_classes":2,"class_mapping":{"0":"Noise/Other","1":f"Target (Class {focus_class})"},"n_folds":int(n_folds),"purge_gap":int(embargo_gap),"n_trees":int(n_trees),"stop_rounds":int(stop_rounds),"max_depth":int(max_depth),"lr":float(lr),"custom_threshold":float(custom_threshold),"fold_pr_aucs":fold_aucs,"fold_auc_mean":auc_mean,"fold_auc_std":auc_std,"final_oos_pr_auc_calibrated":float(final_oos_auc),"is_calibrated":calibrator is not None}
        meta_temp=os.path.join(DATA_DIR,"model_metadata.json")
        with open(meta_temp,'w',encoding='utf-8') as f:
            json.dump(meta_data,f,ensure_ascii=False,indent=4)

        zip_filename=f"HIPO_SniperBrain_{timestamp_str}.zip"
        zip_filepath=os.path.join(DATA_DIR,zip_filename)
        with zipfile.ZipFile(zip_filepath,'w',zipfile.ZIP_DEFLATED) as zf:
            zf.write(master_model_temp,"master_model.json")
            zf.write(meta_temp,"model_metadata.json")
            if calibrator is not None:
                zf.write(calib_temp,"calibrator.json")

        final_msg=(f"### 💎 آموزش تمام شد (FIXED v26 FINAL)\n**PR-AUC فولدها:** {auc_mean:.4f} ± {auc_std:.4f}\n**PR-AUC نهایی OOS (کالیبره):** {final_oos_auc:.5f}\n**تعداد فیچرها:** {len(features)}\n\n**💡 پیشنهاد آستانه (با Wilson CI):** {recommendation_txt}\n\n📋 جدول کامل در HTML گزارش هست.")
        yield final_msg, "\n".join(logs), report_path, zip_filepath

    except Exception:
        err_msg=traceback.format_exc()
        yield "❌ کرش در موتور آموزش", err_msg, None, None

# ==================== HIPO DOCTOR Integrated ====================
def run_doctor_diagnosis(dataset_names, focus_class, n_folds, purge_gap, embargo_gap, max_bars, n_shuffles, progress=gr.Progress()):
    import io
    report_data={"✅":[],"⚠️":[],"❌":[]}
    def verdict(level,text):
        report_data[level].append(text)
        return f"{level} {text}"

    captured=io.StringIO()
    old_stdout=sys.stdout
    sys.stdout=captured

    try:
        if not dataset_names or "No Data" in dataset_names[0]:
            yield "❌ دیتاست انتخاب نشده.", "", None
            return

        print("="*70)
        print("🩺 HIPO DOCTOR v26 FINAL — تشخیص کامل")
        print("="*70)
        print(f"\n📁 دیتاست‌ها: {dataset_names}")
        dfs_raw=[pd.read_parquet(os.path.join(DATA_DIR,d)) for d in dataset_names]
        df_raw=pd.concat(dfs_raw).sort_index()

        print("\n"+"-"*70)
        print("🔬 بخش 1: آلودگی fillna(0) / warm-up")
        print("-"*70)
        nan_frac=df_raw.isna().mean().sort_values(ascending=False)
        top_nan=nan_frac[nan_frac>0.001].head(20)
        if len(top_nan)>0:
            print(top_nan.to_string())
            verdict("⚠️", f"{len(nan_frac[nan_frac>0.001])} ستون دارای NaN → fillna(0) قبلی آلودگی می‌ساخت. در نسخه FIXED با dropna حل شد.")
        else:
            verdict("✅","NaN قابل‌توجهی نیست.")

        df, features = prepare_features_safe(df_raw, ['Target_Class','Signal_Dir','index','level_0','Time','origin_time','origin_price','entry_price','M1_SL','M1_TP','M2_SL','M2_TP','M3_SL','M3_TP','pair','max_bars','Raw_ATR','rev_cross_idx','B_SL','B_TP','S_SL','S_TP','Open','High','Low','Close','Bid'])
        X=df[features].astype('float32')
        y=(df['Target_Class'].astype(int)==int(focus_class)).astype(int)

        n_total=len(df)
        final_idx, calib_start, calib_end, oos_start, safe_embargo = get_safe_splits(n_total, int(max_bars))
        print(f"\nکل نمونه: {n_total} | Base Rate: {y.mean():.3%}")
        print(f"CV: {final_idx} | Calib: {calib_end-calib_start} | Test: {n_total-oos_start}")

        print("\n"+"-"*70)
        print("🔬 بخش 2: کفایت Embargo/Purge vs max_bars")
        print("-"*70)
        if purge_gap < max_bars or embargo_gap < max_bars:
            print(f"purge {purge_gap} / embargo {embargo_gap} < max_bars {max_bars} → نشت!")
            verdict("❌", f"purge/embargo ({purge_gap}/{embargo_gap}) کوچکتر از max_bars ({max_bars}) → نشت لیبل.")
        else:
            verdict("✅", f"purge/embargo ({purge_gap}/{embargo_gap}) نسبت به max_bars ({max_bars}) کافی است. FIXED نسخه از 70 استفاده می‌کند.")
            print(f"purge/embargo کافی: {purge_gap}/{embargo_gap} vs max_bars {max_bars}")

        print("\n"+"-"*70)
        print("🔬 بخش 3: تفاوت نرخ برد Buy vs Sell")
        print("-"*70)
        if 'Signal_Dir' in df_raw.columns:
            buy_mask=df_raw['Signal_Dir']==1
            sell_mask=df_raw['Signal_Dir']==-1
            if buy_mask.sum()>0 and sell_mask.sum()>0:
                yb=y[buy_mask.values]; ys=y[sell_mask.values]
                pb, lob, hib = wilson_ci(yb.sum(), len(yb))
                ps, los, his = wilson_ci(ys.sum(), len(ys))
                print(f"BUY: n={len(yb)} WinRate={pb:.3%} CI {lob:.3%}-{hib:.3%}")
                print(f"SELL: n={len(ys)} WinRate={ps:.3%} CI {los:.3%}-{his:.3%}")
                z,pval=two_prop_ztest(yb.sum(), len(yb), ys.sum(), len(ys))
                print(f"z={z:.2f} p={pval:.4f}")
                if pval<0.05:
                    verdict("❌", f"تفاوت Buy/Sell معنادار p={pval:.4f} → بایاس اسپرد.")
                else:
                    verdict("✅","Buy/Sell تفاوتی ندارد.")
        else:
            verdict("⚠️","Signal_Dir نبود.")

        print("\n"+"-"*70)
        print("🔬 بخش 4: Concept Drift")
        print("-"*70)
        X_cv = X.iloc[:final_idx]
        y_cv = y.iloc[:final_idx]
        thirds=np.array_split(y_cv.values,3)
        for i,ch in enumerate(thirds,1):
            p,lo,hi=wilson_ci(ch.sum(), len(ch))
            print(f"ثلث {i}: n={len(ch)} WinRate={p:.3%} CI {lo:.3%}-{hi:.3%}")
        z,pval=two_prop_ztest(int(thirds[0].sum()), len(thirds[0]), int(thirds[-1].sum()), len(thirds[-1]))
        print(f"اول vs آخر: z={z:.2f} p={pval:.4f}")
        if pval<0.05:
            verdict("⚠️", f"Drift معنادار p={pval:.4f} → رژیم بازار عوض شده.")
        else:
            verdict("✅","Drift دیده نشد.")

        print("\n"+"-"*70)
        print("🔬 بخش 5: ردیف‌های تکراری")
        print("-"*70)
        dup_idx=int(df.index.duplicated().sum())
        dup_rows=int(X.duplicated().sum())
        print(f"ایندکس تکراری: {dup_idx} | ردیف تکراری: {dup_rows}")
        if dup_idx>0:
            verdict("⚠️", f"{dup_idx} ایندکس تکراری → چند سیگنال روی یک کندل.")
        if dup_rows>len(X)*0.02:
            verdict("⚠️", f"{dup_rows} ردیف یکسان >2% → سیگنال دتکتور تکراری.")
        if dup_idx==0 and dup_rows<=len(X)*0.02:
            verdict("✅","تکراری مشکلی نیست.")

        print("\n"+"-"*70)
        print("🔬 بخش 6: آموزش واقعی + اهمیت فیچر")
        print("-"*70)
        base_params=dict(max_depth=5, learning_rate=0.05, tree_method='hist', booster='gbtree', objective='binary:logistic', eval_metric='aucpr', random_state=42, n_jobs=-1)
        tscv=TimeSeriesSplit(n_splits=int(n_folds), gap=safe_embargo)
        real_aucs=[]; last_model=None; last_split=None
        for fold,(tr_idx,te_idx) in enumerate(tscv.split(X_cv),1):
            X_tr=np.ascontiguousarray(X_cv.iloc[tr_idx], dtype=np.float32)
            y_tr=y_cv.iloc[tr_idx].values.astype(int)
            X_te=np.ascontiguousarray(X_cv.iloc[te_idx], dtype=np.float32)
            y_te=y_cv.iloc[te_idx].values.astype(int)
            spw=(y_tr==0).sum()/((y_tr==1).sum()+1e-9)
            m=xgb.XGBClassifier(**base_params, n_estimators=300, scale_pos_weight=spw, early_stopping_rounds=30)
            m.fit(X_tr,y_tr,eval_set=[(X_te,y_te)],verbose=False)
            auc=average_precision_score(y_te, m.predict_proba(X_te)[:,1])
            real_aucs.append(auc)
            last_model,last_split=m,(X_tr,y_tr,X_te,y_te)
            print(f"فولد {fold}: PR-AUC={auc:.4f} Base={y_te.mean():.3%}")
        print(f"میانگین PR-AUC: {np.mean(real_aucs):.4f} ± {np.std(real_aucs):.4f}")
        imp=pd.Series(last_model.feature_importances_, index=features).sort_values(ascending=False)
        print("\nTop10 Importance:\n", imp.head(10).to_string())

        print("\n"+"-"*70)
        print(f"🔬 بخش 7: Label-Shuffle Test ({n_shuffles} تکرار)")
        print("-"*70)
        X_tr,y_tr,X_te,y_te=last_split
        shuffled=[]
        rng=np.random.default_rng(42)
        for i in range(int(n_shuffles)):
            y_tr_shuf=rng.permutation(y_tr)
            spw=(y_tr_shuf==0).sum()/((y_tr_shuf==1).sum()+1e-9)
            m=xgb.XGBClassifier(**base_params, n_estimators=300, scale_pos_weight=spw, early_stopping_rounds=30)
            m.fit(X_tr,y_tr_shuf,eval_set=[(X_te,y_te)],verbose=False)
            shuffled.append(average_precision_score(y_te, m.predict_proba(X_te)[:,1]))
        shuffled=np.array(shuffled)
        real_last=real_aucs[-1]
        p_value=float((shuffled>=real_last).mean())
        print(f"PR-AUC واقعی: {real_last:.4f} | Null mean: {shuffled.mean():.4f} ± {shuffled.std():.4f} | max {shuffled.max():.4f} | p={p_value:.4f}")
        if p_value>0.10:
            verdict("❌", f"مدل از نویز قابل‌تمایز نیست p={p_value:.3f} → مشکل پترن/فیچر است نه مدل.")
        elif p_value>0.03:
            verdict("⚠️", f"مرزی p={p_value:.3f} → سیگنال ضعیف.")
        else:
            verdict("✅", f"مدل بهتر از نویز p={p_value:.3f} → رابطه واقعی هست.")

        print("\n"+"-"*70)
        print("🔬 بخش 8: کالیبراسیون با CI")
        print("-"*70)
        proba=last_model.predict_proba(X_te)[:,1]
        bins=[0,0.3,0.4,0.5,0.6,0.7,0.8,1.01]
        bin_ids=np.digitize(proba,bins)-1
        print(f"{'باکت':<18}{'n':>6}{'WinRate':>12}{'CI Lo':>14}{'CI Hi':>14}")
        for b in range(len(bins)-1):
            mask=bin_ids==b
            n_b=int(mask.sum())
            if n_b==0: continue
            k_b=int(y_te[mask].sum())
            p,lo,hi=wilson_ci(k_b,n_b)
            print(f"{bins[b]:.2f}-{bins[b+1]:.2f}{n_b:>10}{p:>11.2%}{lo:>13.2%}{hi:>13.2%}")
            if n_b<30:
                verdict("⚠️", f"باکت {bins[b]:.2f}-{bins[b+1]:.2f} فقط {n_b} نمونه → عددش بی‌اعتباره.")

        print("\n"+"="*70)
        print("📋 خلاصه نهایی")
        print("="*70)
        for lvl in ["❌","⚠️","✅"]:
            if report_data[lvl]:
                print(f"\n{lvl} ({len(report_data[lvl])}):")
                for i,msg in enumerate(report_data[lvl],1):
                    print(f"  {i}. {msg}")

    except Exception as e:
        print(traceback.format_exc())
        verdict("❌", f"کرش دکتر: {e}")
    finally:
        sys.stdout=old_stdout
        output=captured.getvalue()
        # Build HTML
        html_content = f"""
<!DOCTYPE html><html lang=fa><head><meta charset=UTF-8><title>HIPO DOCTOR FINAL</title>
<style>body{{font-family:monospace;background:#05080f;color:#e0e6ed;padding:20px;direction:rtl;}}h1{{color:#00f2ff;text-align:center;}}pre{{background:#0d1117;padding:15px;border-radius:8px;border:1px solid #333;overflow-x:auto;}} .verdict{{margin-top:15px;padding:10px;border-radius:8px;}} .ok{{border:1px solid #00ff88;background:rgba(0,255,136,0.05);}} .warn{{border:1px solid #ffaa00;background:rgba(255,170,0,0.05);}} .err{{border:1px solid #ff3366;background:rgba(255,51,102,0.08);}}</style></head>
<body><h1>🩺 HIPO DOCTOR v26 FINAL - گزارش کامل</h1>
<h2>خروجی کنسول</h2><pre>{output}</pre>
<h2>خلاصه</h2>
<div class="verdict err"><h3>❌ بحرانی ({len(report_data['❌'])})</h3><ul>{"".join(f"<li>{m}</li>" for m in report_data['❌'])}</ul></div>
<div class="verdict warn"><h3>⚠️ هشدار ({len(report_data['⚠️'])})</h3><ul>{"".join(f"<li>{m}</li>" for m in report_data['⚠️'])}</ul></div>
<div class="verdict ok"><h3>✅ پاس شده ({len(report_data['✅'])})</h3><ul>{"".join(f"<li>{m}</li>" for m in report_data['✅'])}</ul></div>
<div style="margin-top:20px;padding:15px;background:#1a2433;border-radius:8px;"><b>توصیه:</b> اول ❌ ها را فیکس کن، بعد ⚠️ ها. اگر بخش 7 (Label Shuffle) ❌ داد، تمرکز را از تیون مدل بردار و ببر روی پترن/فیچر.</div>
</body></html>
"""
        report_path=os.path.join(DATA_DIR,"HIPO_DOCTOR_FINAL_REPORT.html")
        with open(report_path,"w",encoding='utf-8') as f:
            f.write(html_content)
        yield output, report_path

# ==================== Gradio UI FINAL ====================
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.HTML("<div style='text-align:center;padding:20px;background:#05080f;border-radius:15px;border:1px solid #1a2433;'><h1 style='color:#00ff88;margin:0;font-family:monospace;'>🎯 HIPO AI: SNIPER + DOCTOR v26 FINAL</h1><p style='color:#8b949e;'>Safe Split (embargo=70) | No FillNA | Wilson CI | Doctor Integrated</p></div>")

    with gr.Tabs():
        with gr.TabItem("🎯 آموزش Sniper (FIXED)"):
            with gr.Row():
                with gr.Column(scale=2):
                    w_dataset = gr.CheckboxGroup(choices=get_labeled_files(), label="📁 دیتاست‌های لیبل", value=[get_labeled_files()[0]] if get_labeled_files() else [])
                with gr.Column(scale=1):
                    w_focus = gr.Dropdown(choices=["0","1","2"], label="کلاس هدف", value="1")
                    w_threshold = gr.Slider(minimum=0.3, maximum=0.9, step=0.01, value=0.5, label="آستانه اولیه")
            with gr.Row():
                w_folds = gr.Slider(minimum=3, maximum=10, step=1, value=5, label="فولد (ثبات)")
                w_purge = gr.Slider(minimum=0, maximum=200, step=10, value=70, label="Purge Gap (FIXED: باید >= max_bars=70)")
                w_trees = gr.Slider(minimum=100, maximum=3000, step=100, value=1000, label="تعداد درخت")
                w_stop = gr.Slider(minimum=10, maximum=200, step=10, value=50, label="Early Stop")
                w_depth = gr.Slider(minimum=2, maximum=10, step=1, value=5, label="Max Depth")
                w_lr = gr.Slider(minimum=0.005, maximum=0.3, step=0.005, value=0.03, label="LR")
                w_gpu = gr.Checkbox(label="GPU", value=True)
                w_notes = gr.Textbox(label="یادداشت", value="v26 FINAL - Safe Split + Wilson")
            w_btn = gr.Button("🔥 START CALIBRATED SNIPER TRAINING (FIXED)", variant="primary", size="lg")
            w_msg = gr.Markdown()
            w_log = gr.Textbox(label="لاگ", lines=12)
            w_report = gr.File(label="📄 گزارش HTML Sniper")
            w_zip = gr.File(label="📦 مدل + متادیتا")
            w_dataset.change(update_target_classes, inputs=[w_dataset], outputs=[w_focus])
            w_btn.click(run_sniper_training, inputs=[w_dataset, w_focus, w_folds, w_purge, w_trees, w_stop, w_depth, w_lr, w_gpu, w_notes, w_threshold], outputs=[w_msg, w_log, w_report, w_zip])

        with gr.TabItem("🩺 دکتر HIPO (تشخیص مشکلات)"):
            gr.Markdown("### 🩺 این تب دقیقا همان Doctor شماست، ولی با فیکس‌های جدید (Safe Split + Wilson CI) تا مشکلات واقعی را پیدا کند، نه نویز")
            with gr.Row():
                with gr.Column(scale=2):
                    w_doc_dataset = gr.CheckboxGroup(choices=get_labeled_files(), label="📁 دیتاست برای دکتر", value=[get_labeled_files()[0]] if get_labeled_files() else [])
                with gr.Column(scale=1):
                    w_doc_focus = gr.Dropdown(choices=["0","1","2"], label="کلاس هدف", value="1")
                    w_doc_folds = gr.Slider(minimum=2, maximum=5, step=1, value=2, label="فولد تست")
                    w_doc_purge = gr.Slider(minimum=0, maximum=200, step=10, value=70, label="Purge Gap")
                    w_doc_embargo = gr.Slider(minimum=0, maximum=200, step=10, value=70, label="Embargo Gap")
                    w_doc_max_bars = gr.Slider(minimum=10, maximum=300, step=5, value=60, label="max_bars واقعی لیبل")
                    w_doc_shuffles = gr.Slider(minimum=5, maximum=30, step=1, value=15, label="تعداد Label-Shuffle")
            w_doc_btn = gr.Button("🚀 اجرای دکتر (تشخیص کامل)", variant="secondary", size="lg")
            w_doc_output = gr.Textbox(label="خروجی کنسول دکتر", lines=25)
            w_doc_report = gr.File(label="📄 گزارش HTML دکتر (شامل ✅⚠️❌ و توصیه نهایی)")

            w_doc_btn.click(run_doctor_diagnosis, inputs=[w_doc_dataset, w_doc_focus, w_doc_folds, w_doc_purge, w_doc_embargo, w_doc_max_bars, w_doc_shuffles], outputs=[w_doc_output, w_doc_report])

app.queue().launch(share=True, inbrowser=True)
