from __future__ import annotations
import argparse, json, random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED=42
ID_COLUMNS=["CustomerID"]
LEAKAGE_COLUMNS=["RetentionCalls","RetentionOffersAccepted","MadeCallToRetentionTeam"]

def metrics(y, p, threshold):
    pred=(p>=threshold).astype(int)
    return {"threshold":float(threshold),
            "accuracy":float(accuracy_score(y,pred)),
            "precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)),
            "f1_score":float(f1_score(y,pred,zero_division=0)),
            "roc_auc":float(roc_auc_score(y,p)),
            "average_precision":float(average_precision_score(y,p)),
            "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="cell2celltrain.csv")
    ap.add_argument("--output-dir",default="outputs/cell2cell_ann_threshold_analysis")
    ap.add_argument("--max-epochs",type=int,default=100)
    args=ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data,low_memory=False); df.columns=df.columns.str.strip()
    df=df.drop_duplicates().copy()
    for c in df.select_dtypes(include=["object","string"]).columns:
        df[c]=df[c].map(lambda x:x.strip() if isinstance(x,str) else x).replace("",np.nan)
    target=df["Churn"].astype("string").str.strip().str.lower()
    y=target.map({"yes":1,"no":0,"1":1,"0":0})
    if y.isna().any() or y.nunique()!=2: raise ValueError("Churn must be Yes/No or 1/0.")
    y=y.astype("int64")
    excluded=[c for c in ID_COLUMNS+LEAKAGE_COLUMNS if c in df.columns]
    X=df.drop(columns=["Churn"]+excluded)
    X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=.30,random_state=SEED,stratify=y)
    X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=.5,random_state=SEED,stratify=y_temp)
    numeric=X_train.select_dtypes(include=["number","bool"]).columns.tolist()
    categorical=[c for c in X_train.columns if c not in numeric]
    pre=ColumnTransformer([
        ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric),
        ("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),categorical)
    ],remainder="drop",verbose_feature_names_out=False)
    Xt=pre.fit_transform(X_train).astype("float32")
    Xv=pre.transform(X_val).astype("float32")
    Xs=pre.transform(X_test).astype("float32")
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xt.shape[1],),name="customer_features"),
        tf.keras.layers.Dense(32,activation="relu"),
        tf.keras.layers.Dense(16,activation="relu"),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                           tf.keras.metrics.AUC(name="roc_auc"),
                           tf.keras.metrics.AUC(curve="PR",name="pr_auc")])
    cb=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True,verbose=1)
    hist=model.fit(Xt,y_train.to_numpy(),validation_data=(Xv,y_val.to_numpy()),
                   epochs=args.max_epochs,batch_size=32,callbacks=[cb],verbose=2)
    pv=model.predict(Xv,verbose=0).ravel()
    ps=model.predict(Xs,verbose=0).ravel()
    # Threshold chosen on validation only: maximize F1, tie-break precision, then higher threshold.
    rows=[]
    for t in np.round(np.arange(.05,.951,.01),2):
        m=metrics(y_val.to_numpy(),pv,float(t))
        rows.append({k:v for k,v in m.items() if k!="confusion_matrix"})
    valdf=pd.DataFrame(rows)
    best=valdf.sort_values(["f1_score","precision","threshold"],ascending=[False,False,False]).iloc[0]
    th=float(best["threshold"])
    results={"split_rows":{"train":len(X_train),"validation":len(X_val),"test":len(X_test)},
             "excluded_columns":excluded,"encoded_feature_count":int(Xt.shape[1]),
             "epochs_trained":len(hist.history["loss"]),
             "threshold_selection":"Validation-only threshold grid 0.05-0.95, step 0.01; maximize F1; tie-break precision then higher threshold.",
             "selected_threshold":th,
             "validation_metrics_at_selected_threshold":{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in best.to_dict().items()},
             "test_metrics_at_0.50":metrics(y_test.to_numpy(),ps,.50),
             "test_metrics_at_validation_selected_threshold":metrics(y_test.to_numpy(),ps,th)}
    valdf.to_csv(out/"validation_threshold_analysis.csv",index=False)
    pd.DataFrame({"actual_churn":y_test.to_numpy(),"churn_probability":ps,
                  "predicted_at_050":(ps>=.5).astype(int),"predicted_at_validation_threshold":(ps>=th).astype(int)}).to_csv(out/"test_predictions.csv",index=False)
    model.save(out/"ann_threshold_analysis.keras"); joblib.dump(pre,out/"preprocessor.joblib")
    (out/"threshold_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("\nANN VALIDATION-SELECTED THRESHOLD:",th)
    print("VALIDATION METRICS:",results["validation_metrics_at_selected_threshold"])
    print("TEST METRICS AT 0.50:",results["test_metrics_at_0.50"])
    print("TEST METRICS AT VALIDATION-SELECTED THRESHOLD:",results["test_metrics_at_validation_selected_threshold"])
    print("Saved artifacts:",out.resolve())
if __name__=="__main__": main()
