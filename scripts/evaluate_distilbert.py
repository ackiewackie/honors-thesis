import os
import cv2
import pytesseract
import pandas as pd
import numpy as np
import torch
import joblib
import re

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression

MODEL_NAME = "distilbert-base-uncased"
MODEL_CACHE = "distilbert_logreg_model.pkl"
EMBED_CACHE = "distilbert_train_embeddings.npy"
LABEL_CACHE = "distilbert_train_labels.npy"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DistilBERTEmbedder:
    def __init__(self, model_name=MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=32, max_length=128):
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                last_hidden = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)

                masked_hidden = last_hidden * attention_mask
                summed = masked_hidden.sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                mean_pooled = summed / counts

            embeddings.append(mean_pooled.cpu().numpy())

            print(f"Encoded {min(i + batch_size, len(texts))}/{len(texts)} texts")

        return np.vstack(embeddings)


def load_trained_model():
    if os.path.exists(MODEL_CACHE):
        print("Loading cached classifier...")
        model = joblib.load(MODEL_CACHE)
        embedder = DistilBERTEmbedder()
        return model, embedder

    print("Training classifier from scratch...")
    df = pd.read_csv("data/raw/emails.csv")
    df = df.dropna(subset=["Email Text"])

    texts = df["Email Text"].astype(str).apply(clean_text).tolist()
    labels = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1}).to_numpy()

    embedder = DistilBERTEmbedder()

    if os.path.exists(EMBED_CACHE) and os.path.exists(LABEL_CACHE):
        print("Loading cached training embeddings...")
        X_embed = np.load(EMBED_CACHE)
        y = np.load(LABEL_CACHE)
    else:
        print("Generating training embeddings...")
        X_embed = embedder.encode(texts)
        y = labels
        np.save(EMBED_CACHE, X_embed)
        np.save(LABEL_CACHE, y)

    print("Fitting logistic regression...")
    model = LogisticRegression(max_iter=2000)
    model.fit(X_embed, y)

    joblib.dump(model, MODEL_CACHE)
    print(f"Saved classifier to {MODEL_CACHE}")

    return model, embedder


def classify_email(text, model, embedder):
    text = clean_text(text)

    if not text.strip():
        return 0.0

    text_embed = embedder.encode([text], batch_size=1)
    return model.predict_proba(text_embed)[0][1]


def interpret_risk(score):
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def evaluate_folder(folder_path, model, embedder):
    results = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)

            img = cv2.imread(image_path)
            if img is None:
                print(f"Skipping unreadable file: {filename}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)

            risk_score = classify_email(text, model, embedder)
            risk_band = interpret_risk(risk_score)

            results.append({
                "filename": filename,
                "risk_score": round(risk_score, 3),
                "risk_band": risk_band
            })

            print(f"{filename} → {risk_score:.3f} → {risk_band}")

    return pd.DataFrame(results)


def main():
    model, embedder = load_trained_model()

    folder_path = "data/test_images"
    df_results = evaluate_folder(folder_path, model, embedder)

    df_results.to_csv("evaluation_results_distilbert.csv", index=False)
    print("\nResults saved to evaluation_results_distilbert.csv")


if __name__ == "__main__":
    main()