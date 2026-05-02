"""
app/utils/text_cleaner.py
Utility functions for cleaning and normalizing text.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalize and clean raw text extracted from a resume or job description.

    Steps:
      1. Unicode normalization (NFKD → ASCII-safe)
      2. Remove non-printable control characters
      3. Collapse excessive whitespace
      4. Lowercase (caller can re-uppercase if needed)
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # 2. Remove control characters except newlines and tabs
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)

    # 3. Normalize bullets and dashes
    text = re.sub(r"[•·▪▸●◦‣⁃]", "-", text)

    # 4. Collapse multiple spaces/tabs on the same line
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def extract_sentences(text: str) -> list:
    """Split text into sentences (simple heuristic)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def remove_special_chars(text: str, keep_chars: str = "-.,@+#/") -> str:
    """Remove characters that aren't alphanumeric, spaces, or keep_chars."""
    pattern = r"[^a-zA-Z0-9\s" + re.escape(keep_chars) + r"]"
    return re.sub(pattern, " ", text)
