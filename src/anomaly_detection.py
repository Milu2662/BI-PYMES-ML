import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

FEATURES=["line_count","total_quantity","mean_quantity","mean_unit_price","total_revenue",
          "revenue_abs","min_unit_price","max_unit_price","cancellation_line_share"]

def _matrix(df, feature_cols=None):
    cols=[c for c in (feature_cols or FEATURES) if c in df.columns]
    if not cols: raise ValueError("No hay características disponibles.")
    return df[cols].replace([np.inf,-np.inf],np.nan), cols

def fit_isolation_forest(df, contamination=0.03, random_state=42, feature_cols=None):
    X,cols=_matrix(df,feature_cols)
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",RobustScaler()),
      ("model",IsolationForest(n_estimators=300,contamination=contamination,random_state=random_state,n_jobs=-1))])
    labels=pipe.fit_predict(X); scores=-pipe.decision_function(X)
    out=df.copy(); out["if_label"]=labels; out["if_anomaly"]=(labels==-1).astype(int); out["if_score"]=scores
    return pipe,out,cols

def fit_local_outlier_factor(df, contamination=0.03, n_neighbors=20, feature_cols=None):
    X,cols=_matrix(df,feature_cols); k=max(2,min(int(n_neighbors),len(X)-1))
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",RobustScaler()),
      ("model",LocalOutlierFactor(n_neighbors=k,contamination=contamination,novelty=False,n_jobs=-1))])
    labels=pipe.fit_predict(X); scores=-pipe.named_steps["model"].negative_outlier_factor_
    out=df.copy(); out["lof_label"]=labels; out["lof_anomaly"]=(labels==-1).astype(int); out["lof_score"]=scores
    return pipe,out,cols

def statistical_baseline(df, feature_cols=None, threshold=3.5):
    X,cols=_matrix(df,feature_cols); med=X.median(); mad=(X-med).abs().median().replace(0,np.nan)
    rz=0.6745*(X-med).abs().div(mad); flag=rz.fillna(0).gt(threshold).any(axis=1)
    out=df.copy(); out["baseline_anomaly"]=flag.astype(int); out["baseline_score"]=rz.max(axis=1).fillna(0)
    return out,cols
