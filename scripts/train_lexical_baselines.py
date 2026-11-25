#!/usr/bin/env python3
"""Full-data lexical linear baseline with matched decision losses; no GPU."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, softmax, expit
from scipy.optimize import minimize
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from frontiermem.dual_axis_protocol import ARMS, format_side, load_and_verify
from scripts.eval_weekly_heldout import load_eval_rows, sections_for_source, measure_part, normalize_eval_text
from scripts.train_dual_axis_qwen import LABEL2ID

LABELS=("IGNORE","SUPPORT","DOMINATE")

def text(row, context):
    if not context:
        row={**row,"history":"","context":""}
    return format_side(row)

def pair_text(rows, context):
    return [text(p[k],context) for k in ("positive","negative") for p in rows]

def log_odds(z):
    return logsumexp(z[:,1:],axis=1)-z[:,0]

def score_gradient(z):
    return np.column_stack((-np.ones(len(z)),softmax(z[:,1:],axis=1)))

def binary_loss(z,y):
    s=log_odds(z)
    return np.mean(np.logaddexp(0,s)-y*s), (expit(s)-y)[:,None]*score_gradient(z)/len(z)

def ranking_loss(z):
    s=log_odds(z)
    n=len(z)//2
    v=.5-s[:n]+s[n:]
    g=expit(v)/n
    return np.mean(np.logaddexp(0,v)),np.r_[-g,g][:,None]*score_gradient(z)

def objective(params,xr,yr,xs,arm,lam=.001):
    w=params.reshape(xr.shape[1]+1,3)
    z=xr@w[:-1]+w[-1]
    targets=np.eye(3)[yr]*.95+.05/3
    lp=z-logsumexp(z,axis=1,keepdims=True)
    loss=-np.sum(targets*lp)/len(z)
    dz=(softmax(z,axis=1)-targets)/len(z)
    b,bg=binary_loss(z,yr!=0)
    loss+=.5*b; dz+=.5*bg
    if arm!="pointwise":
        r,rg=ranking_loss(z); loss+=.5*r; dz+=.5*rg
    gw=xr.T@dz; gb=dz.sum(axis=0)
    if arm in ("caid_s2pref","dual_axis"):
        zs=xs@w[:-1]+w[-1]
        n=len(zs)//2
        scale=len(zs)/len(z)
        b,bg=binary_loss(zs,np.r_[np.ones(n),np.zeros(n)])
        loss+=.5*scale*b; ds=.5*scale*bg
        if arm=="dual_axis":
            r,rg=ranking_loss(zs); loss+=.5*scale*r; ds+=.5*scale*rg
        gw+=xs.T@ds; gb+=ds.sum(axis=0)
    loss+=.5*lam*np.sum(w[:-1]**2)
    gw+=lam*w[:-1]
    return float(loss),np.vstack((gw,gb)).ravel()

def prediction(w,x,rows):
    z=x@w[:-1]+w[-1]; scores=log_odds(z); prob=softmax(z,axis=1)
    if not np.isfinite(z).all(): raise RuntimeError("non-finite logits")
    return {r["example_id"]:{"example_id":r["example_id"],"group_id":r["group_id"],
        "gold_binary":r["action"],"score_apply":float(scores[i]),"p_apply":float(prob[i,1:].sum()),
        "prediction_binary":"APPLY" if scores[i]>=0 else "IGNORE",
        "prediction_native":LABELS[int(z[i].argmax())]} for i,r in enumerate(rows)}

def calibration(rows,pred):
    y=np.array([r["action"]=="APPLY" for r in rows],float)
    p=np.array([pred[r["example_id"]]["p_apply"] for r in rows])
    bins=np.minimum((p*10).astype(int),9)
    return {"brier":float(np.mean((p-y)**2)),
            "ece_10_equal_width":float(sum(np.sum(bins==b)/len(y)*abs(p[bins==b].mean()-y[bins==b].mean()) for b in range(10) if np.any(bins==b)))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--context",choices=("full","no_context"),required=True)
    a=p.parse_args(); src=a.source; out=a.out; out.mkdir(parents=True,exist_ok=True)
    d=load_and_verify(src/"data/local/dual_axis_plan/experiment_plan.json",src/"data/external_processed/caid_pairwise_train.jsonl",src/"data/local/s2pref_candidates")
    context=a.context=="full"
    rt,st=d["rpeval_train"],d["s2pref_train"]
    tr=pair_text(rt,context)
    vec=FeatureUnion([("word",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)),
                      ("char",TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=3,max_features=20000,sublinear_tf=True))])
    xr=vec.fit_transform(tr).tocsr()
    xs=vec.transform(pair_text(st,context)).tocsr()
    yr=np.array([LABEL2ID[p[k]["native_label"]] for k in ("positive","negative") for p in rt])
    weights={}
    import joblib
    joblib.dump(vec,out/"vectorizer.joblib")
    for arm in ARMS:
        dest=out/arm; dest.mkdir(exist_ok=True)
        res=minimize(objective,np.zeros((xr.shape[1]+1)*3),args=(xr,yr,xs,arm),
                     method="L-BFGS-B",jac=True,options={"maxiter":200,"ftol":1e-8})
        if not np.isfinite(res.fun): raise RuntimeError("non-finite objective")
        w=res.x.reshape(xr.shape[1]+1,3); weights[arm]=w
        np.save(dest/"weights.npy",w)
        (dest/"optimization.json").write_text(json.dumps({"converged":bool(res.success),"message":str(res.message),"iterations":int(res.nit),"loss":float(res.fun),"features":xr.shape[1],"train_pairs":len(rt),"silver_pairs":len(st) if arm in ("caid_s2pref","dual_axis") else 0},indent=2))
        print("TRAINED",a.context,arm,res.success,res.nit,flush=True)
    # All fitting finished before test sets are loaded; no test-based tuning.
    groups=load_eval_rows(src/"data/external_processed/all_applicability_examples.jsonl")
    for source,rows in groups.items():
        cleaned=[{**r,"history":"" if r.get("history")==r.get("query") else r.get("history","")} for r in rows]
        x=vec.transform([text(r,context) for r in cleaned])
        for arm,w in weights.items():
            pred=prediction(w,x,rows)
            secs=sections_for_source(source,rows)
            scores=[measure_part(rs,pred,source,name) for name,rs in secs.items()]
            scores.append(measure_part(rows,pred,source,source+"_all"))
            dest=out/arm
            (dest/(source+"_metrics.json")).write_text(json.dumps({"model":"tfidf_linear","context":a.context,"arm":arm,"sections":scores,"calibration_fixed":calibration(rows,pred),"silver":"derived_unreviewed","note":"Published benchmark previously examined; decision proxy on BenchPreS."},indent=2))
            with (dest/(source+"_predictions.jsonl")).open("w") as f:
                for value in pred.values(): f.write(json.dumps(value)+"\n")
    print("FULL CPU BASELINES COMPLETE",a.context,flush=True)
if __name__=="__main__": main()
