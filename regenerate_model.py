# ============================================================
#  REGENERATE MODEL + RESULTS  (with WORKING sentiment)
#  Run this ONCE on your machine, in the Fake Review Detector folder.
#
#  FIRST, in a notebook cell, install VADER into THIS Python:
#       %pip install vaderSentiment
#  THEN run this file with the SAME Python your app uses, e.g.:
#       import sys; !"{sys.executable}" regenerate_model.py
#
#  It retrains demo_model/vectorizer/scaler so the app is consistent,
#  and rewrites the results + feature-importance + figures.
# ============================================================
import sys, pandas as pd, numpy as np, re, joblib
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False  # figures are optional; real ones are already in the folder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
from scipy.sparse import hstack, csr_matrix

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _v = SentimentIntensityAnalyzer()
    def sentiment(t): return _v.polarity_scores(t)["compound"]
    print("VADER sentiment: WORKING")
except Exception:
    print("!! VADER NOT INSTALLED - run  %pip install vaderSentiment  first, then rerun this. Aborting.")
    sys.exit(1)

SLANG = ['lekker','sharp sharp','sharp','the plug','no cap','hey','proper','bru','yoh','eish',
         'giving','clean clean','boss','nxa','mooi','best best','shuu','shomafa','shiwa nee']

df = pd.read_csv("namibian_reviews_final.csv"); df["text"] = df["text"].astype(str)
y = df["final_label"].astype(int).values; raw = df["text"].values
def clean_text(t):
    t=t.lower(); t=re.sub(r"http\S+"," ",t); t=re.sub(r"[^a-z0-9\s!]"," ",t); return re.sub(r"\s+"," ",t).strip()
def eng(texts):
    out=[]
    for t in texts:
        low,cc=t.lower(),len(t)
        out.append([len(t.split()),cc,t.count("!"),(sum(1 for c in t if c.isupper())/cc if cc else 0),
                    sum(low.count(s) for s in SLANG),sentiment(t)])
    return np.array(out,dtype=float)
clean=np.array([clean_text(t) for t in raw])

models={"Logistic Regression":lambda:LogisticRegression(max_iter=1000,class_weight="balanced"),
        "Linear SVM":lambda:LinearSVC(class_weight="balanced"),
        "Random Forest":lambda:RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced"),
        "Naive Bayes":lambda:MultinomialNB()}
def run_cv(use):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    res={m:{"acc":[],"prec":[],"rec":[],"f1":[]} for m in models}
    for tr,te in skf.split(clean,y):
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=2000)
        Xtr,Xte=vec.fit_transform(clean[tr]),vec.transform(clean[te])
        if use:
            sc=MinMaxScaler(); Xtr=hstack([Xtr,csr_matrix(sc.fit_transform(eng(raw[tr])))]); Xte=hstack([Xte,csr_matrix(sc.transform(eng(raw[te])))])
        for name,make in models.items():
            clf=make(); clf.fit(Xtr,y[tr]); p=clf.predict(Xte)
            res[name]["acc"].append(accuracy_score(y[te],p)); res[name]["prec"].append(precision_score(y[te],p,zero_division=0))
            res[name]["rec"].append(recall_score(y[te],p,zero_division=0)); res[name]["f1"].append(f1_score(y[te],p,zero_division=0))
    return res
rows=[]
for flag,label in [(False,"TF-IDF only"),(True,"TF-IDF + features")]:
    r=run_cv(flag)
    for name,m in r.items():
        rows.append({"feature_set":label,"model":name,"accuracy":round(np.mean(m["acc"]),3),
                     "precision":round(np.mean(m["prec"]),3),"recall":round(np.mean(m["rec"]),3),"f1":round(np.mean(m["f1"]),3)})
comp=pd.DataFrame(rows); comp.to_csv("results_model_comparison.csv",index=False)
print(comp.to_string(index=False))

vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=2000)
Xall=vec.fit_transform(clean); sc=MinMaxScaler(); Fall=sc.fit_transform(eng(raw))
final=LogisticRegression(max_iter=1000,class_weight="balanced").fit(hstack([Xall,csr_matrix(Fall)]),y)
joblib.dump(final,"demo_model.pkl"); joblib.dump(vec,"demo_vectorizer.pkl"); joblib.dump(sc,"demo_scaler.pkl")

names=["word_count","char_count","exclamations","caps_ratio","slang_count","sentiment"]
fi=pd.DataFrame({"feature":names,"coefficient":[round(float(c),3) for c in final.coef_[0][-6:]]})
fi.to_csv("results_feature_importance.csv",index=False); print(fi.to_string(index=False))

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
op=np.zeros(len(y),dtype=int); opr=np.zeros(len(y))
for tr,te in skf.split(clean,y):
    v=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=2000)
    Xtr,Xte=v.fit_transform(clean[tr]),v.transform(clean[te])
    s=MinMaxScaler(); Xtr=hstack([Xtr,csr_matrix(s.fit_transform(eng(raw[tr])))]); Xte=hstack([Xte,csr_matrix(s.transform(eng(raw[te])))])
    c=LogisticRegression(max_iter=1000,class_weight="balanced").fit(Xtr,y[tr]); op[te]=c.predict(Xte); opr[te]=c.predict_proba(Xte)[:,1]
cm=confusion_matrix(y,op); fpr,tpr,_=roc_curve(y,opr); ra=auc(fpr,tpr)
if HAVE_PLT:
    fig,ax=plt.subplots(figsize=(4.4,4)); ax.imshow(cm,cmap="Blues"); ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Genuine","Fake"]); ax.set_yticklabels(["Genuine","Fake"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Logistic Regression)")
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=14)
    plt.tight_layout(); plt.savefig("confusion_matrix.png",dpi=150); plt.close()
    fig,ax=plt.subplots(figsize=(4.8,4)); ax.plot(fpr,tpr,color="#2563eb",lw=2,label=f"ROC (AUC = {ra:.3f})"); ax.plot([0,1],[0,1],"--",color="#999",lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve (Logistic Regression)"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("roc_curve.png",dpi=150); plt.close()
    print("Figures saved: confusion_matrix.png, roc_curve.png")
else:
    print("(matplotlib not installed - skipped figures; the real ones are already in your folder)")
print("\nConfusion matrix:\n",cm,"\nROC AUC:",round(ra,3))
print("\nDONE - model retrained with working sentiment. Restart the app to use it.")
