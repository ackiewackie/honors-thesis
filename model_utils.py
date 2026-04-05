import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
DATA_PATH = "data/raw/emails.csv"


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Email Text"])

    texts = df["Email Text"].astype(str)
    labels = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})

    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, labels)

    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)

    return model, tfidf


def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(VECTORIZER_PATH)
        return model, tfidf

    return train_and_save_model()


def classify_text(text, model, tfidf):
    if not text.strip():
        return 0.0

    text_tfidf = tfidf.transform([text])
    return model.predict_proba(text_tfidf)[0][1]


def interpret_risk(score):
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    return "LOW"