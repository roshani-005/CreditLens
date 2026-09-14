"""Baseline Logistic Regression and main XGBoost model with calibration and SHAP."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV,calibration_curve
from xgboost import XGBClassifier

def split_data(df,features,seed=42):
    return train_test_split(df[features],df.target.astype(int),test_size=.25,stratify=df.target,random_state=seed)

def train_models(X_train,y_train,X_test,y_test,seed=42):
    prep=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]); Xtr=prep.fit_transform(X_train); Xte=prep.transform(X_test)
    # Logistic Regression is widely useful in lending because coefficients and score contributions are auditable and easier to govern.
    lr=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=seed).fit(Xtr,y_train); lp=lr.predict_proba(Xte)[:,1]
    # Avoid SMOTE: interpolation can create implausible income/debt/behavior combinations and complicate calibration and governance.
    xgb=XGBClassifier(n_estimators=450,max_depth=4,learning_rate=.045,subsample=.82,colsample_bytree=.82,min_child_weight=8,reg_lambda=4,objective="binary:logistic",eval_metric="logloss",scale_pos_weight=max(1.,(y_train==0).sum()/max(1,(y_train==1).sum())),random_state=seed,n_jobs=2).fit(X_train,y_train)
    xp=xgb.predict_proba(X_test)[:,1]
    return {"preprocessor":prep,"logistic":lr,"xgb":xgb,"metrics":{"logistic_regression":{"roc_auc":roc_auc_score(y_test,lp),"pr_auc":average_precision_score(y_test,lp),"brier":brier_score_loss(y_test,lp)},"xgboost":{"roc_auc":roc_auc_score(y_test,xp),"pr_auc":average_precision_score(y_test,xp),"brier":brier_score_loss(y_test,xp)}}}

def calibrate(model,X,y,method="isotonic"):
    return CalibratedClassifierCV(model,method=method,cv=3).fit(X,y)

def plot_calibration(y,raw_p,cal_p,out):
    f1,m1=calibration_curve(y,raw_p,n_bins=10,strategy="quantile"); f2,m2=calibration_curve(y,cal_p,n_bins=10,strategy="quantile"); fig,ax=plt.subplots(figsize=(7,6)); ax.plot(m1,f1,marker="o",label="Raw XGBoost"); ax.plot(m2,f2,marker="o",label="Calibrated"); ax.plot([0,1],[0,1],"--",label="Perfect calibration"); ax.set(xlabel="Predicted probability",ylabel="Observed default rate",title="CreditLens calibration curve"); ax.legend(); fig.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig)

def save_shap(model,X,feature_names,out,max_rows=1500):
    sample=X.sample(min(max_rows,len(X)),random_state=42); sv=shap.TreeExplainer(model).shap_values(sample); sv=sv[1] if isinstance(sv,list) else sv; shap.summary_plot(sv,sample,feature_names=feature_names,show=False,plot_size=(10,7)); plt.tight_layout(); plt.savefig(out,dpi=160,bbox_inches="tight"); plt.close()

def save_bundle(bundle,calibrated,features,out_dir="models"):
    p=Path(out_dir); p.mkdir(exist_ok=True); joblib.dump(bundle["preprocessor"],p/"preprocessor.joblib"); joblib.dump(bundle["logistic"],p/"logistic.joblib"); joblib.dump(bundle["xgb"],p/"xgb.joblib"); joblib.dump(calibrated,p/"calibrated_xgb.joblib"); (p/"features.json").write_text(json.dumps(features,indent=2))
