import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score


def clean_text(text):
    """Basic text cleanup."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label):
    """
    Normalize labels to:
    1 = phishing
    0 = safe
    """
    if pd.isna(label):
        return None

    if isinstance(label, (int, float)):
        if int(label) == 1:
            return 1
        if int(label) == 0:
            return 0

    label_str = str(label).strip().lower()

    phishing_labels = {
        "1", "phishing", "spam", "fraud", "malicious", "unsafe",
        "phishing email"
    }
    safe_labels = {
        "0", "safe", "legitimate", "ham", "benign", "normal",
        "safe email"
    }

    if label_str in phishing_labels:
        return 1
    if label_str in safe_labels:
        return 0

    return None


def infer_label_from_source(filename):
    """
    Fallback labels if a CSV does not contain a usable label column.
    Adjust this if you inspect the files and want different behavior.
    """
    safe_sources = {"Enron.csv"}
    phishing_sources = {
        "CEAS_08.csv",
        "Ling.csv",
        "Nazario.csv",
        "Nigerian_Fraud.csv",
        "SpamAssassin.csv",
        "phishing_email.csv",
    }

    if filename in safe_sources:
        return 0
    if filename in phishing_sources:
        return 1
    return None


def find_text_columns(df):
    """
    Return text-related columns that actually exist in the dataframe.
    """
    possible_columns = [
        "subject", "body", "sender", "urls", "receiver",
        "text", "text_combined", "email", "content", "message",
        "Email Text"
    ]
    return [col for col in possible_columns if col in df.columns]


def build_text_column(df):
    """
    Combine useful fields into one text column.
    Uses only columns that exist.
    """
    existing_columns = find_text_columns(df)

    if not existing_columns:
        raise ValueError(
            "None of the expected text columns were found. "
            f"Available columns: {list(df.columns)}"
        )

    combined = []
    for _, row in df.iterrows():
        parts = []
        for col in existing_columns:
            parts.append(clean_text(row[col]))
        combined.append(" ".join(parts).strip())

    df["text"] = combined
    return df, existing_columns


def load_and_combine_csvs(dataset_path):
    """
    Load all CSV files in the downloaded dataset folder and combine them.
    Uses each file's label column when present; otherwise falls back
    to source-based labeling.
    """
    csv_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".csv")])
    csv_files = [f for f in csv_files if f != "emails.csv"]


    if not csv_files:
        raise FileNotFoundError("No CSV files found in the downloaded dataset.")

    print("\nCSV files found:")
    for f in csv_files:
        print(" -", f)

    all_dfs = []

    for filename in csv_files:
        file_path = os.path.join(dataset_path, filename)
        print(f"\nLoading {filename}...")

        df = pd.read_csv(file_path)
        df["source_file"] = filename

        print("Columns:", df.columns.tolist())

        df, used_columns = build_text_column(df)
        print("Text columns used:", used_columns)

        if "label" in df.columns:
            df["label"] = df["label"].apply(normalize_label)
            print("Used existing label column.")
        elif "Email Type" in df.columns:
            df["label"] = df["Email Type"].apply(normalize_label)
            print("Used 'Email Type' as label column.")
        else:
            fallback_label = infer_label_from_source(filename)
            if fallback_label is None:
                print(f"Skipping {filename}: no label column and no fallback label rule.")
                continue
            df["label"] = fallback_label
            print(f"Inferred label from source file: {fallback_label}")

        df = df[["text", "label", "source_file"]].copy()
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No usable CSV files were loaded into the combined dataset.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def main():
    dataset_path = "data/raw" 

    print("Loading dataset from:", dataset_path)

    df = load_and_combine_csvs(dataset_path)

    before_drop = len(df)
    df = df.dropna(subset=["label", "text"]).copy()
    df = df[df["text"].str.strip() != ""].copy()
    after_drop = len(df)

    df["label"] = df["label"].astype(int)

    print(f"\nRows before cleaning: {before_drop}")
    print(f"Rows after cleaning:  {after_drop}")

    print("\nOverall label distribution:")
    print(df["label"].value_counts())

    print("\nRows by source file:")
    print(df["source_file"].value_counts())

    # Optional: save the combined dataset
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/emails_metadata_enhanced.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved combined dataset to: {output_path}")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(model, "saved_models/enhanced_logreg_model.pkl")
    joblib.dump(vectorizer, "saved_models/enhanced_tfidf_vectorizer.pkl")

    print("\nSaved model to: saved_models/enhanced_logreg_model.pkl")
    print("Saved vectorizer to: saved_models/enhanced_tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()