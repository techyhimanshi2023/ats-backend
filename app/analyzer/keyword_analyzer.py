"""
app/analyzer/keyword_analyzer.py

Two-stage keyword matching:
  Stage 1 (Exact)    : string/regex match — fast, precise
  Stage 2 (Semantic) : BERT sentence-transformers — catches synonyms/paraphrases

If sentence-transformers is not installed, falls back gracefully to exact-only.
"""

import re


class KeywordAnalyzer:
    """
    Hybrid exact + semantic keyword analyzer.

    Results include:
      matched          : exact matches
      semantic_matched : additional matches found via BERT (not exact)
      missing          : keywords found in neither exact nor semantic pass
      match_pct        : combined match percentage
      semantic_score   : overall BERT similarity score (0–100)
    """

    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "this", "that", "these", "those", "we", "you", "he", "she",
        "it", "they", "our", "your", "their", "us", "its",
        "as", "if", "so", "not", "no", "nor", "yet", "both",
        "all", "any", "each", "few", "more", "most", "other",
        "some", "such", "into", "than", "then", "when", "where",
        "who", "which", "about", "above", "also", "only", "very",
        "can", "must", "need", "use", "using", "used",
    }

    def __init__(self, use_bert: bool = True, bert_threshold: float = 42.0):
        self.use_bert = use_bert
        self.bert_threshold = bert_threshold
        self._bert = None

    def analyze(self, resume_text: str, job_keywords: list) -> dict:
        resume_lower = resume_text.lower()
        resume_tokens = set(self._tokenize(resume_lower))

        # Stage 1: Exact matching
        exact_matched = []
        not_exact = []
        for kw in job_keywords:
            kw_lower = kw.lower()
            hit = (kw_lower in resume_lower) if " " in kw_lower else (kw_lower in resume_tokens)
            (exact_matched if hit else not_exact).append(kw)

        # Stage 2: BERT semantic matching on leftovers
        semantic_matched = []
        missing = not_exact
        semantic_score = 0.0
        semantic_details = {}

        if self.use_bert and not_exact:
            bert_result = self._run_bert_semantic(resume_text, not_exact)
            semantic_matched = bert_result.get("semantic_matched", [])
            missing = bert_result.get("semantic_missing", [])
            semantic_details = bert_result.get("match_details", {})

        if self.use_bert:
            semantic_score = self._bert_overall_score(resume_text, job_keywords)

        total = len(job_keywords)
        total_matched = len(exact_matched) + len(semantic_matched)
        match_pct = (total_matched / total * 100) if total > 0 else 0.0

        return {
            "matched": exact_matched,
            "semantic_matched": semantic_matched,
            "missing": missing,
            "total_job_keywords": total,
            "match_pct": round(match_pct, 2),
            "semantic_score": round(semantic_score, 1),
            "semantic_match_details": semantic_details,
            "keyword_density": self._compute_density(resume_lower, exact_matched),
        }

    def _get_bert(self):
        if self._bert is None:
            try:
                from app.ml.bert_similarity import BERTSimilarity
                self._bert = BERTSimilarity()
            except ImportError:
                self.use_bert = False
                return None
        return self._bert

    def _run_bert_semantic(self, resume_text: str, keywords: list) -> dict:
        bert = self._get_bert()
        if bert is None:
            return {"semantic_matched": [], "semantic_missing": keywords, "match_details": {}}
        try:
            return bert.semantic_keyword_match(resume_text, keywords, threshold=self.bert_threshold)
        except Exception as e:
            print(f"  [BERT] Warning: {e}. Falling back to exact only.")
            return {"semantic_matched": [], "semantic_missing": keywords, "match_details": {}}

    def _bert_overall_score(self, resume_text: str, keywords: list) -> float:
        bert = self._get_bert()
        if bert is None or not keywords:
            return 0.0
        try:
            return bert.analyze(resume_text, " ".join(keywords)).get("semantic_score", 0.0)
        except Exception:
            return 0.0

    def _tokenize(self, text: str) -> list:
        tokens = re.findall(r"\b[a-z][a-z0-9+#.]*\b", text)
        return [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]

    def _compute_density(self, text: str, keywords: list) -> dict:
        return {kw: len(re.findall(re.escape(kw.lower()), text)) for kw in keywords}
