# @title 🔬 HIPO FEATURE↔LABEL QUALITY LAB [v1.0] { display-mode: "form" }
# بعد از Labeling و قبل از Training اجرا شود.
# خروجی: Global Score، KEEP/DROP، selected_features.json، گزارش HTML

import sys, subprocess
def smart_install():
    reqs=["gradio","pyarrow","matplotlib","scikit-learn","pandas","numpy"]
    miss=[]
    for r in reqs:
        mod="sklearn" if r=="scikit-learn" else r.replace("-","_")
        try: __import__(mod)
        except ImportError: miss.append(r)
    if miss:
        subprocess.check_call([sys.executable,"-m","pip","install","-q"]+miss)
smart_install()

import os, glob, json, warnings
from datetime import datetime
import numpy as np, pandas as pd, gradio as gr, matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")

DATA_DIR="/content/hipo_lab_data"
os.makedirs(DATA_DIR, exist_ok=True)
DROP={"Open","High","Low","Close","Volume","Tick_Up_Count","Tick_Down_Count","Bid","Ask","Target_Class","Signal_Dir","entry_price","M1_SL","M1_TP","max_bars","pair","Raw_ATR"}

def get_labeled():
    fs=glob.glob(os.path.join(DATA_DIR,"*_Labeled.parquet"))
    return sorted({os.path.basename(f).replace("_Labeled.parquet","") for f in fs}) or ["No Labeled Data Found"]

def is_num(s):
    return bool(pd.api.types.is_numeric_dtype(s))

def prepare_xy(df):
    y=df["Target_Class"].astype(int).values
    cols=[c for c in df.columns if c not in DROP and not str(c).startswith("__") and is_num(df[c])]
    X=df[cols].replace([np.inf,-np.inf],np.nan)
    return X,y,cols

def score_features(X,y,cols):
    Xm=X.fillna(X.median(numeric_only=True))
    try: mi=mutual_info_classif(Xm[cols].values.astype(np.float64),y,random_state=42,n_neighbors=min(5,max(1,len(y)//20)))
    except Exception: mi=np.zeros(len(cols))
    rows=[]
    for i,c in enumerate(cols):
        v=Xm[c].values.astype(np.float64)
        if np.std(v)<1e-12: auc,r=0.5,0.0
        else:
            order=np.argsort(v); yo=y[order]; np_=float(yo.sum()); nn=float(len(yo)-np_)
            if np_<1 or nn<1: auc=0.5
            else:
                ranks=np.arange(1,len(yo)+1,dtype=np.float64)
                auc=(ranks[yo==1].sum()-np_*(np_+1)/2)/(np_*nn); auc=max(auc,1-auc)
            r=float(np.corrcoef(v,y)[0,1]); r=0.0 if np.isnan(r) else r
        q=0.45*np.clip((auc-0.5)/0.15,0,1)*100 + 0.35*np.clip(mi[i]/0.12,0,1)*100 + 0.20*np.clip(abs(r)/0.25,0,1)*100
        verdict="KEEP_STRONG" if q>=55 else ("KEEP" if q>=35 else ("WEAK" if q>=20 else "DROP"))
        rows.append(dict(feature=c,AUC_abs=round(float(auc),4),MI=round(float(mi[i]),5),abs_corr=round(abs(r),4),quality_0_100=round(float(q),2),verdict=verdict))
    return pd.DataFrame(rows).sort_values("quality_0_100",ascending=False).reset_index(drop=True)

def select_features(feat_df,X,max_f=35,min_q=25.0):
    cands=feat_df[feat_df.quality_0_100>=min_q]
    if len(cands)==0: cands=feat_df.head(max_f)
    sel=[]
    for _,row in cands.iterrows():
        c=row.feature
        if c not in X.columns or len(sel)>=max_f: 
            if len(sel)>=max_f: break
            continue
        v=X[c].fillna(X[c].median()).values; ok=True
        for s in sel:
            vs=X[s].fillna(X[s].median()).values
            if np.std(v)<1e-12 or np.std(vs)<1e-12: continue
            rr=abs(np.corrcoef(v,vs)[0,1])
            if not np.isnan(rr) and rr>=0.92: ok=False; break
        if ok: sel.append(c)
    return sel

def probe(X,y,sel,n_folds=4):
    if len(sel)==0 or len(y)<30: return dict(cv_pr_auc=0,base=float(y.mean()) if len(y) else 0,lift=0,shuffle_p=1,real=False)
    Xm=X[sel].fillna(X[sel].median()).astype(np.float32); base=float(y.mean())
    tscv=TimeSeriesSplit(n_splits=min(n_folds,max(2,len(y)//25)),gap=5)
    sc=[]
    for tr,te in tscv.split(Xm):
        if y[tr].sum()<3 or y[te].sum()<2 or (1-y[tr]).sum()<3: continue
        clf=HistGradientBoostingClassifier(max_depth=3,max_iter=80,learning_rate=0.08,min_samples_leaf=max(5,len(tr)//40),l2_regularization=1.0,random_state=42)
        clf.fit(Xm.iloc[tr],y[tr]); sc.append(average_precision_score(y[te],clf.predict_proba(Xm.iloc[te])[:,1]))
    if not sc: return dict(cv_pr_auc=0,base=base,lift=0,shuffle_p=1,real=False)
    real=float(np.mean(sc)); rng=np.random.default_rng(42); sh=[]
    for _ in range(12):
        ys=y.copy(); rng.shuffle(ys); fs=[]
        for tr,te in tscv.split(Xm):
            if ys[tr].sum()<3 or ys[te].sum()<2: continue
            clf=HistGradientBoostingClassifier(max_depth=3,max_iter=50,learning_rate=0.08,min_samples_leaf=max(5,len(tr)//40),l2_regularization=1.0,random_state=0)
            clf.fit(Xm.iloc[tr],ys[tr]); fs.append(average_precision_score(ys[te],clf.predict_proba(Xm.iloc[te])[:,1]))
        if fs: sh.append(float(np.mean(fs)))
    p=(1+sum(1 for s in sh if s>=real-1e-12))/(len(sh)+1) if sh else 1.0
    lift=real/(base+1e-9)
    return dict(cv_pr_auc=round(real,4),cv_std=round(float(np.std(sc)),4),base=round(base,4),lift=round(lift,3),shuffle_p=round(float(p),4),real=bool(p<0.15 and lift>1.08))

def global_score(feat_df,pr,n,wr):
    uni=float(np.clip(feat_df.head(10).quality_0_100.mean()/70,0,1)*35)
    multi=30*float(np.clip((pr.get("lift",1)-1)/0.5,0,1)); 
    if not pr.get("real"): multi*=0.5
    sample=10*float(np.clip(np.log1p(n)/np.log1p(2000),0,1))
    wrs=10*float(np.clip(1-abs(wr-0.45)/0.45,0,1))
    strong=int((feat_df.verdict=="KEEP_STRONG").sum()); cover=15*float(np.clip(strong/10,0,1))
    return round(float(np.clip(uni+multi+sample+wrs+cover,0,100)),2)

def run_lab(datasets,max_f,min_q,n_folds,save_filt,progress=gr.Progress()):
    if not datasets or "No Labeled" in str(datasets[0]): return "❌ لیبل نیست",None,None,None,None
    frames=[]
    for name in datasets:
        p=os.path.join(DATA_DIR,f"{name}_Labeled.parquet")
        if os.path.exists(p):
            d=pd.read_parquet(p); d["__ds__"]=name; frames.append(d)
    if not frames: return "❌ فایل نیست",None,None,None,None
    df=pd.concat(frames).sort_index(); n=len(df); wr=float((df.Target_Class==1).mean())
    X,y,cols=prepare_xy(df); feat_df=score_features(X,y,cols)
    sel=select_features(feat_df,X,int(max_f),float(min_q))
    for _,r in feat_df.iterrows():
        if str(r.feature).startswith(("PSA_","LSW_","FVG_","PS_")) and r.verdict in ("KEEP","KEEP_STRONG") and r.feature not in sel and len(sel)<int(max_f)+5:
            sel.append(r.feature)
    pr=probe(X,y,sel,int(n_folds)); score=global_score(feat_df,pr,n,wr)
    if score>=65 and pr.get("real"):
        interp="فیچرها با لیبل هم‌راستا و سیگنال واقعی‌اند. می‌توانید Training را شروع کنید."
        action="Sniper با selected_features.json"
    elif score>=45:
        interp="کیفیت متوسط. DROPها را حذف و GA لیبل/فیچر v25 را تقویت کنید."
        action="Feature v25 + GA multi-obj + دوباره Quality Lab"
    else:
        interp="فیچر↔لیبل ضعیف. Training نکنید."
        action="اول نمونه و فیچرهای PSA/LSW را بسازید"
    ts=datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html=os.path.join(DATA_DIR,f"HIPO_FeatureQuality_{ts}.html")
    color="#00ff88" if score>=60 else ("#ffcc00" if score>=40 else "#ff3366")
    rows="".join(f"<tr><td>{r.feature}</td><td>{r.AUC_abs}</td><td>{r.MI}</td><td>{r.quality_0_100}</td><td>{r.verdict}</td></tr>" for _,r in feat_df.head(40).iterrows())
    open(html,"w",encoding="utf-8").write(f"""<html><body style='background:#0b0f19;color:#eee;font-family:monospace;padding:20px'>
    <h1>Feature↔Label Quality</h1><div style='font-size:64px;color:{color}'>{score}</div>
    <p>n={n} WR={wr*100:.1f}% selected={len(sel)} shuffle_p={pr.get('shuffle_p')} cvPR={pr.get('cv_pr_auc')} lift={pr.get('lift')}</p>
    <p>{interp}</p><p>{action}</p>
    <table border=1 cellpadding=4>{rows}</table></body></html>""")
    sel_path=os.path.join(DATA_DIR,"selected_features.json")
    json.dump({"selected_features":sel,"global_score":score,"probe":pr,"feat_table":feat_df.head(80).to_dict("records"),
               "summary":{"n":n,"wr":wr,"interp":interp,"action":action}}, open(sel_path,"w"), ensure_ascii=False, indent=2, default=str)
    if save_filt:
        keep=["Open","High","Low","Close","Volume","Target_Class","Signal_Dir","entry_price","M1_SL","M1_TP","max_bars"]+sel
        for name in datasets:
            sub=df[df["__ds__"]==name] if "__ds__" in df.columns else df
            cols=[c for c in keep if c in sub.columns]
            sub[cols].to_parquet(os.path.join(DATA_DIR,f"{name}_Labeled_FeatFiltered.parquet"), compression="snappy")
    figp=os.path.join(DATA_DIR,f"FQ_groups_{ts}.png")
    # simple top bar
    fig,ax=plt.subplots(figsize=(9,4),facecolor="#0b0f19"); ax.set_facecolor("#0b0f19")
    top=feat_df.head(15); ax.barh(top.feature[::-1], top.quality_0_100[::-1], color="#00f2ff"); ax.tick_params(colors="white")
    ax.set_title("Top feature quality", color="#00ff88"); fig.tight_layout(); fig.savefig(figp,dpi=120); plt.close()
    msg=f"""### 🔬 Quality Lab Result
**Global Score: `{score}/100`**
- Samples: **{n}** | WR: **{wr*100:.1f}%** | Selected: **{len(sel)}**
- CV PR-AUC: **{pr.get('cv_pr_auc')}** | Lift: **{pr.get('lift')}×** | Shuffle p: **{pr.get('shuffle_p')}** → {'✅ REAL' if pr.get('real') else '⚠️ weak'}
- {interp}
- **Action:** {action}
- saved: `{sel_path}`
"""
    return msg, feat_df.head(40), figp, html, sel_path

with gr.Blocks(title="HIPO Feature Quality Lab") as app:
    gr.HTML("<h1 style='color:#00f2ff;text-align:center'>🔬 FEATURE ↔ LABEL QUALITY LAB</h1>")
    gr.Markdown("اگر Score < 45 است Training را شروع نکنید.")
    with gr.Row():
        w_data=gr.CheckboxGroup(choices=get_labeled(), value=[get_labeled()[0]] if get_labeled() else [], label="Labeled datasets")
        w_max=gr.Slider(10,80,value=35,step=1,label="max features")
        w_min=gr.Slider(10,60,value=25,step=1,label="min quality")
        w_folds=gr.Slider(2,6,value=4,step=1,label="CV folds")
        w_save=gr.Checkbox(value=True,label="save FeatFiltered parquet")
    btn=gr.Button("🚀 RUN QUALITY LAB", variant="primary")
    msg=gr.Markdown(); top=gr.DataFrame(); fig=gr.Image(type="filepath"); html=gr.File(); js=gr.File()
    btn.click(run_lab, inputs=[w_data,w_max,w_min,w_folds,w_save], outputs=[msg,top,fig,html,js])
app.queue().launch(share=True, inbrowser=True)
