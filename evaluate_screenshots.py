import os
import cv2
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_trained_model():
    df = pd.read_csv('data/raw/emails.csv')
    df = df.dropna(subset=['Email Text'])

    texts = df['Email Text']
    labels = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})

    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X_tfidf, labels)

    return model, tfidf

def classify_email(text, model, tfidf):
    if not text.strip():
        return 0.0

    text_tfidf = tfidf.transform([text])
    return model.predict_proba(text_tfidf)[0][1]

def interpret_risk(score):
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def evaluate_folder(folder_path, model, tfidf):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)

            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)

            risk_score = classify_email(text, model, tfidf)
            risk_band = interpret_risk(risk_score)

            results.append({
                "filename": filename,
                "risk_score": round(risk_score, 3),
                "risk_band": risk_band
            })

            print(f"{filename} → {risk_score:.3f} → {risk_band}")

    return pd.DataFrame(results)

def main():
    model, tfidf = load_trained_model()

    folder_path = "data/test_images"
    df_results = evaluate_folder(folder_path, model, tfidf)

    df_results.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")

if __name__ == "__main__":
    main()