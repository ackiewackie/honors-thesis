import cv2
import pytesseract
from PIL import Image
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time

def load_trained_model():
    """Load the trained phishing classifier"""
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
    """Return phishing probability score"""
    if not text.strip():
        return 0.0

    text_tfidf = tfidf.transform([text])
    return model.predict_proba(text_tfidf)[0][1]

def process_email_image(image_path, model, tfidf):
    """Extract text from image and classify"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)
    risk_score = classify_email(text, model, tfidf)

    return text, risk_score

def interpret_risk(risk_score):
    """Map score to risk band"""
    if risk_score >= 0.8:
        return "🚨 HIGH RISK: Likely phishing"
    elif risk_score >= 0.5:
        return "⚠️  MEDIUM RISK: Review recommended"
    else:
        return "✅ LOW RISK: Likely safe"

def continuous_monitoring(image_path, model, tfidf, interval=2):
    """Simulate continuous email monitoring"""
    print("Starting continuous email monitoring...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            text, risk_score = process_email_image(image_path, model, tfidf)

            print("\n--- OCR Output (first 200 chars) ---")
            print(text[:200])
            print(f"\nPhishing Risk Score: {risk_score:.3f}")
            print(interpret_risk(risk_score))

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def main():
    print("Loading phishing classifier...")
    model, tfidf = load_trained_model()

    image_path = "data/email_sample.png"

    print(f"\nProcessing email image: {image_path}")
    text, risk_score = process_email_image(image_path, model, tfidf)

    print("\n--- Extracted Text ---")
    print(text)
    print(f"\nPhishing Risk Score: {risk_score:.3f}")
    print(interpret_risk(risk_score))

    # Uncomment to enable continuous monitoring
    # continuous_monitoring(image_path, model, tfidf)

if __name__ == "__main__":
    main()
