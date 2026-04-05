import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import collections
from datetime import datetime

def load_data():
    print("Loading data...")
    df = pd.read_csv('data/raw/emails.csv')
    print(f"Loaded {len(df)} rows")
    df = df.dropna(subset=['Email Text'])
    print(f"After removing empty emails: {len(df)} rows")
    texts = df['Email Text']
    labels = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    print("Data loading complete")
    return texts, labels

def main():
    texts, labels = load_data()
    
    # Class distribution
    print("\nClass distribution:")
    print(collections.Counter(labels))
    print()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    print("\nFitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    print("Transforming test data...")
    X_test_tfidf = tfidf.transform(X_test)
    
    print("Training logistic regression model...")
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    print("Making predictions...")
    preds = model.predict(X_test_tfidf)
    
    print("Calculating metrics...")
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    print("\nSaving results...")
    # Save results
    with open('results/baseline_results.txt', 'w') as f:
        f.write(f"Dataset: Phishing Email Detection\n")
        f.write(f"Model: TF-IDF + Logistic Regression\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"F1 Score: {f1:.3f}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("Results saved to results/baseline_results.txt")

if __name__ == "__main__":
    main()