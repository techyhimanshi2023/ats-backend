"""
app/ml/bert_similarity.py

Uses sentence-transformers (BERT-based) to compute semantic similarity
between resume text and job description.

Instead of exact keyword matching, this compares meaning — so
"built neural networks" can match "deep learning experience required".

Model used: all-MiniLM-L6-v2 (fast, lightweight, ~80MB)
"""

from __future__ import annotations
import os
import numpy as np

# Lazy imports — only loaded when BERTSimilarity is actually used
_sentence_transformer = None


def _load_model(model_name: str):
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  [BERT] Loading model '{model_name}' (first run may download ~80MB)...")
            _sentence_transformer = SentenceTransformer(model_name)
            print("  [BERT] Model loaded.")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
    return _sentence_transformer


class BERTSimilarity:
    """
    Computes semantic similarity between resume chunks and job description
    using a pre-trained sentence-transformer model.

    Usage:
        bert = BERTSimilarity()
        result = bert.analyze(resume_text, job_text)
        print(result["semantic_score"])   # 0.0 – 100.0
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # lazy load

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, resume_text: str, job_text: str) -> dict:
        """
        Compare full resume against full job description semantically.

        Returns:
            dict:
                semantic_score      : float 0–100 (overall similarity)
                sentence_scores     : list of (sentence, score) top matches
                top_resume_sentences: top 5 resume sentences most relevant to JD
        """
        model = self._get_model()

        resume_sentences = self._split_sentences(resume_text)
        job_sentences    = self._split_sentences(job_text)

        if not resume_sentences or not job_sentences:
            return {"semantic_score": 0.0, "sentence_scores": [], "top_resume_sentences": []}

        # Encode
        resume_embeddings = model.encode(resume_sentences, convert_to_numpy=True)
        job_embeddings    = model.encode(job_sentences,    convert_to_numpy=True)

        # Overall similarity: mean of max similarities per job sentence
        similarity_matrix = self._cosine_similarity_matrix(resume_embeddings, job_embeddings)

        # For each job sentence, find the best matching resume sentence
        best_per_job = similarity_matrix.max(axis=0)   # shape: (num_job_sentences,)
        overall_sim  = float(np.mean(best_per_job))

        # Top resume sentences (most relevant to the JD overall)
        resume_relevance = similarity_matrix.max(axis=1)  # shape: (num_resume_sentences,)
        top_indices = np.argsort(resume_relevance)[::-1][:5]
        top_sentences = [
            {"sentence": resume_sentences[i], "score": round(float(resume_relevance[i]) * 100, 1)}
            for i in top_indices
        ]

        # Score on 0–100 scale
        # Cosine similarity is -1 to 1; typical range for good matches: 0.3–0.9
        # Rescale: (sim - 0.0) / 1.0 * 100, clamp 0–100
        semantic_score = round(min(max(overall_sim * 100, 0.0), 100.0), 1)

        return {
            "semantic_score": semantic_score,
            "top_resume_sentences": top_sentences,
            "raw_similarity": round(overall_sim, 4),
        }

    def sentence_pair_score(self, text_a: str, text_b: str) -> float:
        """
        Simple cosine similarity between two text snippets. Returns 0–100.
        Useful for comparing a single skill/sentence pair.
        """
        model = self._get_model()
        emb_a = model.encode([text_a], convert_to_numpy=True)
        emb_b = model.encode([text_b], convert_to_numpy=True)
        sim = self._cosine_similarity_matrix(emb_a, emb_b)[0, 0]
        return round(float(min(max(sim * 100, 0.0), 100.0)), 1)

    def semantic_keyword_match(self, resume_text: str, keywords: list, threshold: float = 40.0) -> dict:
        """
        For each keyword in the list, check if the resume contains a
        semantically similar phrase (even if the exact word is absent).

        Args:
            resume_text : full resume text
            keywords    : list of keywords from job description
            threshold   : minimum similarity score (0–100) to count as matched

        Returns:
            dict:
                semantic_matched : keywords matched semantically
                semantic_missing : keywords still not found
                match_details    : per-keyword best score
        """
        if not keywords:
            return {"semantic_matched": [], "semantic_missing": [], "match_details": {}}

        model = self._get_model()
        resume_sentences = self._split_sentences(resume_text)

        if not resume_sentences:
            return {"semantic_matched": [], "semantic_missing": keywords, "match_details": {}}

        resume_embeddings = model.encode(resume_sentences, convert_to_numpy=True)
        keyword_embeddings = model.encode(keywords, convert_to_numpy=True)

        # similarity_matrix shape: (num_keywords, num_resume_sentences)
        sim_matrix = self._cosine_similarity_matrix(keyword_embeddings, resume_embeddings)
        best_scores = sim_matrix.max(axis=1) * 100  # per keyword

        matched = []
        missing = []
        details = {}

        for kw, score in zip(keywords, best_scores):
            score_f = round(float(score), 1)
            details[kw] = score_f
            if score_f >= threshold:
                matched.append(kw)
            else:
                missing.append(kw)

        return {
            "semantic_matched": matched,
            "semantic_missing": missing,
            "match_details": details,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    @staticmethod
    def _split_sentences(text: str, min_len: int = 10) -> list:
        """Split text into non-trivial sentences."""
        import re
        raw = re.split(r"(?<=[.!?\n])\s+", text)
        return [s.strip() for s in raw if len(s.strip()) >= min_len]

    @staticmethod
    def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between every row of a and every row of b.
        Returns matrix of shape (len(a), len(b)).
        """
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return np.dot(a_norm, b_norm.T)
