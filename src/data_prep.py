"""Lending Club ingestion plus a synthetic Indian-fintech feature layer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["annual_inc","loan_amnt","int_rate","installment","dti","fico_range_low","fico_range_high","emp_length_num","pub_rec","inq_last_6mths","open_acc","revol_util","delinq_2yrs","term_months","home_ownership_code","salary_credit_regularity","upi_txn_frequency_30d","upi_txn_frequency_60d","upi_txn_frequency_90d","upi_txn_volume_trend","recharge_frequency","bill_payment_punctuality_score","emi_to_income_ratio","obligation_to_income_ratio","pincode_income_proxy"]

# Synthetic feature rationale:
# salary_credit_regularity = stability of recurring salary-credit timing.
# UPI frequencies = digital transaction activity at 30/60/90-day horizons.
# UPI volume trend = improving/deteriorating transaction activity.
# recharge_frequency = recurring mobile-payment behavior proxy.
# bill_payment_punctuality_score = simulated bill-payment discipline.
# EMI/obligation-to-income = affordability and payment-burden measures.
# pincode_income_proxy = synthetic socioeconomic proxy used ONLY for fairness auditing.

def _num(s):
    return pd.to_numeric(s.astype(str).str.replace("%","",regex=False), errors="coerce")

def _target(df):
    if "bad_loan" in df:
        return pd.to_numeric(df.bad_loan, errors="coerce").map({1:1,2:0})
    status=df.get("loan_status",pd.Series(index=df.index,dtype=object)).astype(str).str.lower()
    y=pd.Series(np.nan,index=df.index)
    y[status.str.contains("charged off|default|late",regex=True,na=False)]=1
    y[status.str.contains("fully paid",regex=True,na=False)]=0
    return y

def _base_features(df):
    out=pd.DataFrame(index=df.index)
    cols=["annual_inc","loan_amnt","int_rate","installment","dti","fico_range_low","fico_range_high","pub_rec","inq_last_6mths","open_acc","revol_util","delinq_2yrs"]
    for c in cols: out[c]=_num(df[c]) if c in df else np.nan
    emp=df.get("emp_length",pd.Series("< 1 year",index=df.index)).astype(str)
    out["emp_length_num"]=pd.to_numeric(emp.str.extract(r"(\d+)")[0],errors="coerce").fillna(0).clip(0,10)
    term=df.get("term",pd.Series("36 months",index=df.index)).astype(str)
    out["term_months"]=pd.to_numeric(term.str.extract(r"(\d+)")[0],errors="coerce").fillna(36)
    home=df.get("home_ownership",pd.Series("OTHER",index=df.index)).astype(str)
    out["home_ownership_code"]=pd.Categorical(home).codes.astype(float)
    return out

def add_synthetic_india_features(X,seed=42):
    rng=np.random.default_rng(seed); n=len(X)
    annual=X.annual_inc.fillna(X.annual_inc.median()).clip(lower=10000); monthly=annual/12
    loan=X.loan_amnt.fillna(X.loan_amnt.median()).clip(lower=500)
    inst=X.installment.fillna(loan/36).clip(lower=20)
    X["salary_credit_regularity"]=np.clip(rng.gamma(2.5,1.6,n)+.000002*annual,.2,12)
    X["upi_txn_frequency_30d"]=np.clip(rng.poisson(18+np.sqrt(annual/50000),n),0,120).astype(float)
    X["upi_txn_frequency_60d"]=X.upi_txn_frequency_30d+rng.poisson(16,n)
    X["upi_txn_frequency_90d"]=X.upi_txn_frequency_60d+rng.poisson(15,n)
    X["upi_txn_volume_trend"]=np.clip(.08-.35*(inst/monthly).clip(0,2)+rng.normal(0,.12,n),-.9,1.5)
    X["recharge_frequency"]=np.clip(rng.normal(24,6,n)+3*(inst/monthly),5,60)
    delay=np.clip(rng.gamma(1.5,2.2,n)+10*(inst/monthly).clip(0,1),0,45)
    X["bill_payment_punctuality_score"]=np.clip(100-2.2*delay+rng.normal(0,3,n),0,100)
    X["emi_to_income_ratio"]=np.clip(inst/monthly,0,2)
    debt=X.dti.fillna(18).clip(0,80)/100
    X["obligation_to_income_ratio"]=np.clip(.55*debt+.65*X.emi_to_income_ratio,0,2)
    rank=pd.Series(annual).rank(pct=True).to_numpy()
    X["pincode_income_proxy"]=np.clip(np.floor(10+80*rank+rng.normal(0,8,n)),1,99)
    return X

def load_and_prepare(path,sample_n=None,seed=42):
    df=pd.read_csv(Path(path),compression="infer",low_memory=False,nrows=sample_n)
    X=add_synthetic_india_features(_base_features(df),seed); X["target"]=_target(df)
    X=X.dropna(subset=["target"]).reset_index(drop=True)
    for c in FEATURES: X[c]=pd.to_numeric(X[c],errors="coerce")
    return X

def make_demo_lending_club(n=12000,seed=42):
    rng=np.random.default_rng(seed); annual=np.exp(rng.normal(np.log(65000),.65,n)).clip(18000,400000); loan=np.exp(rng.normal(np.log(14000),.65,n)).clip(1000,50000); inst=loan*(.025+rng.uniform(.001,.012,n)); dti=np.clip(rng.normal(18,9,n),0,60); fico=np.clip(rng.normal(700,45,n),600,850)
    X=pd.DataFrame({"annual_inc":annual,"loan_amnt":loan,"int_rate":np.clip(7+(700-fico)*.08+rng.normal(0,2,n),5,30),"installment":inst,"dti":dti,"fico_range_low":fico-5,"fico_range_high":fico+5,"pub_rec":rng.poisson(.2,n),"inq_last_6mths":rng.poisson(1.1,n),"open_acc":rng.poisson(8,n)+1,"revol_util":np.clip(rng.normal(48,25,n),0,120),"delinq_2yrs":rng.poisson(.4,n),"emp_length_num":rng.integers(0,11,n),"term_months":rng.choice([36,60],n,p=[.72,.28]),"home_ownership_code":rng.integers(0,4,n).astype(float)})
    X=add_synthetic_india_features(X,seed+1); logit=-3+.055*(dti-18)+.06*(X.obligation_to_income_ratio*100-35)-.018*(fico-700)+.025*(X.int_rate-12)-.015*(X.bill_payment_punctuality_score-75)-.012*X.upi_txn_volume_trend*100+rng.normal(0,.45,n); p=1/(1+np.exp(-logit)); X["target"]=rng.binomial(1,np.clip(p,.01,.9)); return X

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--sample",type=int,default=250000); ap.add_argument("--output",default="data/processed/creditlens_features.csv"); a=ap.parse_args(); out=load_and_prepare(a.input,a.sample); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f"Saved {len(out):,} rows -> {a.output}")
