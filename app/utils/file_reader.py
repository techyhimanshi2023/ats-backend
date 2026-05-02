"""
app/utils/file_reader.py
Utility helpers for reading files safely.
"""

import os


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """Read a plain text file safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()


def get_file_extension(path: str) -> str:
    """Return lowercase extension without dot."""
    return os.path.splitext(path)[1].lower().lstrip(".")


def is_supported_resume_format(path: str) -> bool:
    return get_file_extension(path) in ("pdf", "docx", "doc", "txt")
