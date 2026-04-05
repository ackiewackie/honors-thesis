import os
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


MODEL_DIR = "saved_models"

# -- Model Selection ----------------------------------------------------------
# To use the BASELINE model, keep this block uncommented:
#MODEL_PATH        = os.path.join(MODEL_DIR, "phishing_model.pkl")
#VECTORIZER_PATH   = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
#ACTIVE_MODEL_NAME = "Baseline"
# -----------------------------------------------------------------------------

# To use the ENHANCED model instead, comment out the block above and
# uncomment this block:
MODEL_PATH        = os.path.join(MODEL_DIR, "enhanced_logreg_model.pkl")
VECTORIZER_PATH   = os.path.join(MODEL_DIR, "enhanced_tfidf_vectorizer.pkl")
ACTIVE_MODEL_NAME = "Enhanced"
# -----------------------------------------------------------------------------

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
        return model, tfidf, ACTIVE_MODEL_NAME

    model, tfidf = train_and_save_model()
    return model, tfidf, ACTIVE_MODEL_NAME


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


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EnhancedPhishingDetector:
    def __init__(
        self,
        model_path="saved_models/enhanced_logreg_model.pkl",
        vectorizer_path="saved_models/enhanced_tfidf_vectorizer.pkl"
    ):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        cleaned = clean_text(text)
        text_vec = self.vectorizer.transform([cleaned])

        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0][1]

        return {
            "label": int(prediction),
            "phishing_probability": float(probability)
        }