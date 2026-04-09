import os
import re
import numpy as np
import pandas as pd
import torch
import joblib

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH   = "data/raw/emails.csv"
MODEL_NAME  = "distilbert-base-uncased"
SAVE_DIR    = "saved_models"
MODEL_CACHE = os.path.join(SAVE_DIR, "distilbert_logreg_model.pkl")
EMBED_CACHE = os.path.join(SAVE_DIR, "distilbert_train_embeddings.npy")
LABEL_CACHE = os.path.join(SAVE_DIR, "distilbert_train_labels.npy")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DistilBERTEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=32, max_length=128):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=max_length, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = self.model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy())
            print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")
        return np.vstack(embeddings)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Email Text"])
    texts  = df["Email Text"].astype(str).apply(clean_text).tolist()
    labels = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1}).to_numpy()

    print(f"Total samples: {len(texts)}")

    embedder = DistilBERTEmbedder()

    if os.path.exists(EMBED_CACHE) and os.path.exists(LABEL_CACHE):
        print("Loading cached embeddings...")
        X = np.load(EMBED_CACHE)
        y = np.load(LABEL_CACHE)
    else:
        print("Generating embeddings (this may take a while)...")
        X = embedder.encode(texts)
        y = labels
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.save(EMBED_CACHE, X)
        np.save(LABEL_CACHE, y)
        print(f"Embeddings cached to {EMBED_CACHE}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    print("Training logistic regression...")
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_CACHE)
    print(f"Model saved to {MODEL_CACHE}")

    y_pred = model.predict(X_test)

    print("\n=== DistilBERT + Logistic Regression Results ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
