"""
app/ml/ats_predictor.py

Logistic Regression model that predicts the probability of a resume
passing ATS screening for a given job description.

Feature engineering:
  - TF-IDF on combined (resume + job description) text
  - Handcrafted features: keyword overlap %, skills overlap %, years exp

Training:
  Run  python train.py  after placing your CSV in training_data/

Inference:
  predictor = ATSPredictor()
  predictor.load()
  result = predictor.predict(resume_text, job_text, keyword_pct, skills_pct, years)
"""

from __future__ import annotations
import os
import pickle
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/lr_ats_model.pkl")
MODEL_PATH = os.path.normpath(MODEL_PATH)


class ATSPredictor:
    """
    Logistic Regression ATS pass/fail predictor.

    Attributes:
        model    : trained sklearn LogisticRegression
        vectorizer: fitted TfidfVectorizer
        is_loaded: bool
    """

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_loaded = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, training_data_path: str, save_path: str = MODEL_PATH) -> dict:
        """
        Train the model from a CSV file.

        CSV format (with header):
            resume_text,job_description,label
            "Full resume text...","Job description text...",1
            "Another resume...","Another JD...",0

        label: 1 = passed ATS / got interview, 0 = rejected

        Args:
            training_data_path : path to the CSV
            save_path          : where to save the trained model

        Returns:
            dict with accuracy, precision, recall, f1, n_samples
        """
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import scipy.sparse as sp

        print(f"  [LR] Loading training data from: {training_data_path}")
        df = pd.read_csv(training_data_path)

        required_cols = {"resume_text", "job_description", "label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV must have columns: {required_cols}\n"
                f"Found: {set(df.columns)}"
            )

        df = df.dropna(subset=["resume_text", "job_description", "label"])
        df["label"] = df["label"].astype(int)

        print(f"  [LR] {len(df)} samples loaded. Label distribution:")
        print(f"       Pass (1): {df['label'].sum()}  |  Fail (0): {(df['label']==0).sum()}")

        # --- Feature engineering ---
        # 1. TF-IDF on combined text
        combined_texts = (df["resume_text"] + " [SEP] " + df["job_description"]).tolist()

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf_features = vectorizer.fit_transform(combined_texts)

        # 2. Handcrafted features
        handcrafted = self._build_handcrafted_features(df)
        X = sp.hstack([tfidf_features, sp.csr_matrix(handcrafted)])
        y = df["label"].values

        # --- Train / test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- Model ---
        model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        model.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"\n  [LR] Training complete.")
        print(f"       Accuracy : {acc:.2%}")
        print(f"       Precision: {report['1']['precision']:.2%}")
        print(f"       Recall   : {report['1']['recall']:.2%}")
        print(f"       F1       : {report['1']['f1-score']:.2%}")

        # --- Save ---
        self.model = model
        self.vectorizer = vectorizer
        self.is_loaded = True

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"model": model, "vectorizer": vectorizer}, f)
        print(f"  [LR] Model saved → {save_path}")

        return {
            "accuracy": round(acc, 4),
            "precision": round(report["1"]["precision"], 4),
            "recall": round(report["1"]["recall"], 4),
            "f1": round(report["1"]["f1-score"], 4),
            "n_samples": len(df),
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def load(self, model_path: str = MODEL_PATH) -> bool:
        """
        Load a previously saved model.
        Returns True if successful, False if model file doesn't exist yet.
        """
        if not os.path.exists(model_path):
            return False
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.vectorizer = data["vectorizer"]
        self.is_loaded = True
        return True

    def predict(
        self,
        resume_text: str,
        job_text: str,
        keyword_match_pct: float = 0.0,
        skills_match_pct: float = 0.0,
        years_experience: int = 0,
    ) -> dict:
        """
        Predict ATS pass probability for a resume + job description pair.

        Args:
            resume_text       : cleaned resume text
            job_text          : job description text
            keyword_match_pct : keyword overlap percentage (0–100)
            skills_match_pct  : skills overlap percentage (0–100)
            years_experience  : years of experience found in resume

        Returns:
            dict:
                pass_probability : float 0–100 (probability of passing ATS)
                prediction       : "PASS" or "FAIL"
                confidence       : "High" / "Medium" / "Low"
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. Run train.py first, then call predictor.load()."
            )

        import scipy.sparse as sp

        combined = resume_text + " [SEP] " + job_text
        tfidf_feat = self.vectorizer.transform([combined])

        import re
        def token_overlap(text_a, text_b):
            a = set(re.findall(r"\b\w+\b", str(text_a).lower()))
            b = set(re.findall(r"\b\w+\b", str(text_b).lower()))
            return len(a & b) / len(b) if b else 0.0

        handcrafted = np.array([[
            token_overlap(resume_text, job_text),
            min(len(str(resume_text).split()) / 500.0, 1.0),
        ]])

        X = sp.hstack([tfidf_feat, sp.csr_matrix(handcrafted)])
        prob = self.model.predict_proba(X)[0]

        pass_prob = round(float(prob[1]) * 100, 1)
        prediction = "PASS" if pass_prob >= 50 else "FAIL"

        if pass_prob >= 75 or pass_prob <= 25:
            confidence = "High"
        elif pass_prob >= 60 or pass_prob <= 40:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "pass_probability": pass_prob,
            "prediction": prediction,
            "confidence": confidence,
        }

    def is_model_trained(self) -> bool:
        return os.path.exists(MODEL_PATH)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_handcrafted_features(df) -> np.ndarray:
        """
        Build a matrix of handcrafted numeric features from the dataframe.
        Columns: keyword_overlap, skill_overlap, resume_length_norm
        """
        import re

        def token_overlap(text_a, text_b):
            a = set(re.findall(r"\b\w+\b", str(text_a).lower()))
            b = set(re.findall(r"\b\w+\b", str(text_b).lower()))
            if not b:
                return 0.0
            return len(a & b) / len(b)

        def resume_length_norm(text):
            return min(len(str(text).split()) / 500.0, 1.0)

        rows = []
        for _, row in df.iterrows():
            rows.append([
                token_overlap(row["resume_text"], row["job_description"]),
                resume_length_norm(row["resume_text"]),
                # label-independent proxy features only
            ])
        return np.array(rows)
