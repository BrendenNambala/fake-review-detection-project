"""
Fake Review Detector - prediction API (FastAPI).
Serves the trained model over HTTP so the browser extension can classify reviews.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
Then open http://127.0.0.1:8000/docs to test.

Files that must sit next to this main.py:
    demo_model.pkl, demo_vectorizer.pkl, demo_scaler.pkl
"""
import re
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# IMPORTANT: keep identical to the slang list the model was trained with (your 19 words).
SLANG = ['lekker', 'sharp sharp', 'sharp', 'the plug', 'no cap', 'hey', 'proper', 'bru', 'yoh', 'eish', 'giving', 'clean clean', 'boss', 'nxa', 'mooi', 'best best', 'shuu', 'shomafa', 'shiwa nee']

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _v = SentimentIntensityAnalyzer()
    def sentiment(t): return _v.polarity_scores(t)["compound"]
except Exception:
    def sentiment(t): return 0.0

def clean_text(t):
    t = t.lower(); t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s!]", " ", t); return re.sub(r"\s+", " ", t).strip()

def eng_features(text):
    low, cc = text.lower(), len(text)
    return np.array([[len(text.split()), cc, text.count("!"),
        (sum(1 for c in text if c.isupper()) / cc if cc else 0),
        sum(low.count(s) for s in SLANG), sentiment(text)]], dtype=float)

model = joblib.load("demo_model.pkl")
vec = joblib.load("demo_vectorizer.pkl")
scaler = joblib.load("demo_scaler.pkl")
try:
    FEAT_NAMES = list(vec.get_feature_names_out())
except Exception:
    FEAT_NAMES = list(vec.get_feature_names())

def predict_one(text):
    X = vec.transform([clean_text(text)])
    F = scaler.transform(eng_features(text))
    prob = float(model.predict_proba(hstack([X, csr_matrix(F)]))[:, 1][0])
    # evidence words
    row = vec.transform([clean_text(text)]).tocoo()
    coef = model.coef_[0]
    contribs = [(FEAT_NAMES[i], float(v * coef[i])) for i, v in zip(row.col, row.data) if i < len(FEAT_NAMES)]
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    evidence = [{"word": w, "direction": "fake" if c >= 0 else "genuine"} for w, c in contribs[:8]]
    return {
        "verdict": "LIKELY FAKE" if prob >= 0.5 else "LIKELY GENUINE",
        "fake_probability": round(prob, 3),
        "evidence": evidence,
    }

app = FastAPI(title="Fake Review Detector API", version="1.0")
# Allow the extension (any origin) to call the API. Fine for a research prototype.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ReviewIn(BaseModel):
    text: str

class BatchIn(BaseModel):
    texts: list[str]

@app.get("/")
def health():
    return {"status": "ok", "service": "Fake Review Detector API"}

@app.post("/predict")
def predict(item: ReviewIn):
    if not item.text or not item.text.strip():
        return {"error": "empty text"}
    return predict_one(item.text)

@app.post("/predict_batch")
def predict_batch(item: BatchIn):
    return {"results": [predict_one(t) for t in item.texts if t and t.strip()]}
