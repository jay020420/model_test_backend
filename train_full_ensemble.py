
# -*- coding: utf-8 -*-
"""
train_full_ensemble.py
Usage:
    python train_full_ensemble.py --root "/Users/llouis/Documents/model_test" --k 3 --topq 0.10
Creates:
    data/preds.csv  (pred_xgb, pred_lgbm, pred_rf, pred_gb, pred_dl)
    risk_output_trained.csv
"""
import os, argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

def read_csv_smart(path):
    for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV 인코딩 해석 실패: {path}")

def to_month(s):
    dt = pd.to_datetime(s.astype(str), errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").astype(str))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--topq", type=float, default=0.10)
    args = ap.parse_args()

    BASE_DIR = args.root
    DATA_DIR = os.path.join(BASE_DIR, "data")

    import sys
    sys.path.insert(0, BASE_DIR)
    from pipeline import run_pipeline

    ds1 = read_csv_smart(os.path.join(DATA_DIR, "big_data_set1_f.csv"))
    ds2 = read_csv_smart(os.path.join(DATA_DIR, "ds2_monthly_usage.csv"))
    ds3 = read_csv_smart(os.path.join(DATA_DIR, "ds3_monthly_customers.csv"))

    KEY_MCT, KEY_YM = "ENCODED_MCT", "TA_YM"

    def build_labels_robust(ds1, ds2, ds3, k_months=3, topq=0.10):
        df = ds2.merge(ds3, on=[KEY_MCT, KEY_YM], how="outer")
        df[KEY_YM] = to_month(df[KEY_YM]); df[KEY_MCT] = df[KEY_MCT].astype(str)
        df = df.sort_values([KEY_MCT, KEY_YM]).reset_index(drop=True); df["y"]=0
        if "MCT_ME_D" in ds1.columns:
            tmp = ds1[[KEY_MCT, "MCT_ME_D"]].copy()
            tmp[KEY_MCT]=tmp[KEY_MCT].astype(str); tmp["MCT_ME_D"]=pd.to_datetime(tmp["MCT_ME_D"], errors="coerce")
            df = df.merge(tmp, on=KEY_MCT, how="left")
            t0=df[KEY_YM]; tK=t0+pd.offsets.MonthEnd(0)+pd.DateOffset(months=k_months)
            cond=(df["MCT_ME_D"].notna()) & (df["MCT_ME_D"]>t0) & (df["MCT_ME_D"]<=tK)
            df.loc[cond,"y"]=1
        if df["y"].nunique()<2:
            def bin2num(s):
                s=s.astype(str); m=s.str.extract(r"(\\d+)", expand=False)
                return pd.to_numeric(m, errors="coerce")
            df["RC_SAA_num"]=bin2num(df.get("RC_M1_SAA",""))
            df["RC_CUS_num"]=bin2num(df.get("RC_M1_UE_CUS_CN",""))
            df["dSAA"]=df.groupby(KEY_MCT)["RC_SAA_num"].diff()
            df["dCUS"]=df.groupby(KEY_MCT)["RC_CUS_num"].diff()
            cxl=pd.to_numeric(df.get("APV_CE_RAT",0), errors="coerce")
            indme=pd.to_numeric(df.get("M12_SME_RY_ME_MCT_RAT",0), errors="coerce")
            bznme=pd.to_numeric(df.get("M12_SME_BZN_ME_MCT_RAT",0), errors="coerce")
            sig=(df["dSAA"]<=-10).astype(int)+(df["dCUS"]<=-10).astype(int)+(cxl>=90).astype(int)+(indme>=80).astype(int)+(bznme>=80).astype(int)
            df["y_proxy"]=(sig>=2).astype(int)
            if df["y_proxy"].nunique()>=2 and df["y_proxy"].sum()>0: df["y"]=df["y_proxy"]
        if df["y"].nunique()<2:
            out = run_pipeline(ds1, ds2, ds3, preds=None)
            outj = out.merge(df[[KEY_MCT, KEY_YM]], on=[KEY_MCT, KEY_YM], how="right")
            pf = pd.to_numeric(outj["p_final"], errors="coerce").fillna(0)
            thr = pf.quantile(1-topq); df["y"]=(pf>=thr).astype(int)
        return df

    robust_df = build_labels_robust(ds1, ds2, ds3, k_months=args.k, topq=args.topq)

    num_cols = [
        "M1_SME_RY_SAA_RAT","M1_SME_RY_CNT_RAT",
        "M12_SME_RY_SAA_PCE_RT","M12_SME_BZN_SAA_PCE_RT",
        "M12_SME_RY_ME_MCT_RAT","M12_SME_BZN_ME_MCT_RAT",
        "DLV_SAA_RAT","MCT_UE_CLN_REU_RAT","MCT_UE_CLN_NEW_RAT"
    ]
    cat_cols = [c for c in ["HPSN_MCT_ZCD_NM","HPSN_MCT_BZN_CD_NM"] if c in robust_df.columns]

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score

    X=robust_df[num_cols+cat_cols].copy(); y=robust_df["y"].astype(int)
    num_transform=Pipeline([("imp", SimpleImputer(strategy="median"))])
    ct=ColumnTransformer([("num",num_transform,num_cols),("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)], remainder="drop")

    # Imports for models
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import xgboost as xgb, lightgbm as lgb
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y if y.nunique()>1 else None, test_size=0.25, random_state=42)

    rf = Pipeline([("prep", ct), ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"))])
    rf.fit(Xtr, ytr); prf=rf.predict_proba(Xte)[:,1]

    gb = Pipeline([("prep", ct), ("clf", GradientBoostingClassifier(random_state=42))])
    gb.fit(Xtr, ytr); pgb=gb.predict_proba(Xte)[:,1]

    xgb_clf = Pipeline([("prep", ct), ("clf", xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, tree_method="hist"))])
    xgb_clf.fit(Xtr,ytr); pxgb=xgb_clf.predict_proba(Xte)[:,1]

    lgb_clf = Pipeline([("prep", ct), ("clf", lgb.LGBMClassifier(
        n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        objective="binary", random_state=42))])
    lgb_clf.fit(Xtr, ytr); plgb=lgb_clf.predict_proba(Xte)[:,1]

    Xd_tr = ct.fit_transform(Xtr); Xd_te = ct.transform(Xte)
    inp = keras.Input(shape=(Xd_tr.shape[1],)); h=layers.Dense(128, activation="relu")(inp); h=layers.Dropout(0.2)(h)
    h=layers.Dense(64, activation="relu")(h); outp=layers.Dense(1, activation="sigmoid")(h)
    dl_model = keras.Model(inp,outp); dl_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    dl_model.fit(Xd_tr, ytr, epochs=10, batch_size=256, verbose=0); pdl=dl_model.predict(Xd_te, verbose=0).ravel()

    def metrics(p): return {"roc_auc": float(roc_auc_score(yte,p)), "pr_auc": float(average_precision_score(yte,p))}
    print("RF",metrics(prf)); print("GB",metrics(pgb)); print("XGB",metrics(pxgb)); print("LGB",metrics(plgb)); print("DL",metrics(pdl))

    from sklearn.linear_model import LogisticRegression
    w={"xgb":0.25,"lgb":0.25,"rf":0.25,"gb":0.15,"dl":0.10}
    stack = w["xgb"]*pxgb + w["lgb"]*plgb + w["rf"]*prf + w["gb"]*pgb + w["dl"]*pdl
    pl = LogisticRegression(max_iter=200); pl.fit(stack.reshape(-1,1), yte)
    print("Ensemble(cal) AUC:", roc_auc_score(yte, pl.predict_proba(stack.reshape(-1,1))[:,1]))

    # Full predict
    def predict_full(Xf):
        prf_f  = rf.predict_proba(Xf)[:,1]
        pgb_f  = gb.predict_proba(Xf)[:,1]
        pxgb_f = xgb_clf.predict_proba(Xf)[:,1]
        plgb_f = lgb_clf.predict_proba(Xf)[:,1]
        Xd_full = ct.transform(Xf); pdl_f = dl_model.predict(Xd_full, verbose=0).ravel()
        stack_f = w["xgb"]*pxgb_f + w["lgb"]*plgb_f + w["rf"]*prf_f + w["gb"]*pgb_f + w["dl"]*pdl_f
        pcal_f  = pl.predict_proba(stack_f.reshape(-1,1))[:,1]
        return prf_f, pgb_f, pxgb_f, plgb_f, pdl_f, pcal_f

    prf_f, pgb_f, pxgb_f, plgb_f, pdl_f, pcal_f = predict_full(X)

    preds_full = robust_df[[ "ENCODED_MCT", "TA_YM" ]].copy()
    preds_full["ENCODED_MCT"]=preds_full["ENCODED_MCT"].astype(str)
    preds_full["TA_YM"]=to_month(preds_full["TA_YM"])
    preds_full["pred_xgb"]=pxgb_f; preds_full["pred_lgbm"]=plgb_f
    preds_full["pred_rf"]=prf_f; preds_full["pred_gb"]=pgb_f; preds_full["pred_dl"]=pdl_f
    preds_full = preds_full.dropna(subset=["ENCODED_MCT","TA_YM"])
    preds_path = os.path.join(DATA_DIR, "preds.csv"); preds_full.to_csv(preds_path, index=False, encoding="utf-8")
    print("Saved preds:", preds_path)

    # pipeline 재실행 (타입 강제)
    p = pd.read_csv(preds_path)
    p["ENCODED_MCT"]=p["ENCODED_MCT"].astype(str)
    p["TA_YM"]=to_month(p["TA_YM"]); p=p.dropna(subset=["ENCODED_MCT","TA_YM"])
    out = run_pipeline(ds1, ds2, ds3, preds=p)
    out_path = os.path.join(BASE_DIR, "risk_output_trained.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
