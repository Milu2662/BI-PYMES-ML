import numpy as np, pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from .anomaly_detection import fit_isolation_forest,fit_local_outlier_factor,statistical_baseline

def inject_synthetic_anomalies(df,fraction=.03,random_state=42):
    rng=np.random.default_rng(random_state); out=df.copy().reset_index(drop=True); n=len(out)
    idx=rng.choice(n,size=max(1,min(n,int(round(n*fraction)))),replace=False)
    out["synthetic_anomaly_truth"]=0; out.loc[idx,"synthetic_anomaly_truth"]=1
    cols=[c for c in ["total_quantity","mean_unit_price","total_revenue","line_count","revenue_abs"] if c in out]
    for i in idx:
        c=rng.choice(cols); val=pd.to_numeric(pd.Series([out.at[i,c]]),errors="coerce").iloc[0]
        if pd.isna(val): val=1.0
        factor=rng.choice([8.0,-6.0,10.0]); out.at[i,c]=float(val)*factor if val!=0 else float(factor)
    return out

def _metrics(y,p,name):
    cm=confusion_matrix(y,p,labels=[0,1])
    return {"model":name,"precision":precision_score(y,p,zero_division=0),
      "recall":recall_score(y,p,zero_division=0),"f1":f1_score(y,p,zero_division=0),
      "tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1])}

def run_comparison(df,contamination=.03,random_state=42):
    data=inject_synthetic_anomalies(df,contamination,random_state)
    _,a,_=fit_isolation_forest(data,contamination,random_state)
    _,b,_=fit_local_outlier_factor(data,contamination)
    c,_=statistical_baseline(data); y=data.synthetic_anomaly_truth.to_numpy()
    metrics=pd.DataFrame([_metrics(y,a.if_anomaly.to_numpy(),"Isolation Forest"),
      _metrics(y,b.lof_anomaly.to_numpy(),"Local Outlier Factor"),
      _metrics(y,c.baseline_anomaly.to_numpy(),"Robust Z-score")])
    return data,a,b,c,metrics
