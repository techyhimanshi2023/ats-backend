"""
app/parser/resume_parser.py
Parses PDF and DOCX resume files into structured data.
"""

import os
import re


class ResumeParser:
    """
    Parses a resume file (PDF or DOCX) and extracts:
      - raw_text       : full text of the resume
      - name           : candidate name (best guess)
      - email          : email address
      - phone          : phone number
      - skills         : list of detected skills
      - education      : education entries
      - experience     : work experience entries
      - sections_found : which standard sections were detected
    """

    SECTION_HEADERS = [
        "education", "experience", "work experience", "employment",
        "skills", "technical skills", "projects", "certifications",
        "summary", "objective", "profile", "achievements", "awards",
        "publications", "languages", "interests", "hobbies", "references"
    ]

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: str) -> dict:
        """
        Parse a resume file and return structured data.

        Args:
            file_path: Absolute or relative path to .pdf or .docx file.

        Returns:
            dict with keys: raw_text, name, email, phone, skills,
                            education, experience, sections_found.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            raw_text = self._extract_pdf(file_path)
        elif ext in (".docx", ".doc"):
            raw_text = self._extract_docx(file_path)
        elif ext == ".txt":
            raw_text = self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")

        return self._structure(raw_text, file_path)

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_pdf(self, path: str) -> str:
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            pass

        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            pass

        try:
            import subprocess
            result = subprocess.run(
                ["pdftotext", path, "-"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass

        raise RuntimeError(
            "Cannot extract PDF text. Install pdfplumber: pip install pdfplumber"
        )

    def _extract_docx(self, path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(para.text for para in doc.paragraphs)
        except ImportError:
            raise RuntimeError(
                "python-docx not installed. Run: pip install python-docx"
            )

    def _extract_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Structuring
    # ------------------------------------------------------------------

    def _structure(self, raw_text: str, file_path: str) -> dict:
        lines = raw_text.splitlines()

        return {
            "raw_text": raw_text,
            "file_path": file_path,
            "name": self._extract_name(lines),
            "email": self._extract_email(raw_text),
            "phone": self._extract_phone(raw_text),
            "skills": self._extract_skills(raw_text),
            "education": self._extract_section(lines, ["education"]),
            "experience": self._extract_section(
                lines, ["experience", "work experience", "employment"]
            ),
            "sections_found": self._detect_sections(raw_text),
        }

    def _extract_name(self, lines: list) -> str:
        """Heuristic: the first non-empty short line is usually the name."""
        for line in lines[:10]:
            line = line.strip()
            if 2 <= len(line.split()) <= 5 and not re.search(r"[@|/\\]", line):
                if not any(h in line.lower() for h in self.SECTION_HEADERS):
                    return line
        return ""

    def _extract_email(self, text: str) -> str:
        match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> str:
        match = re.search(
            r"(\+?\d[\d\s\-().]{7,}\d)", text
        )
        return match.group(0).strip() if match else ""

    def _extract_skills(self, text: str) -> list:
        """
        Very broad skill extraction based on a curated keyword list.
        Returns unique lowercased skills found in the resume.
        """
        KNOWN_SKILLS = [
            # Programming languages
            "python", "java", "javascript", "typescript", "c++", "c#", "c",
            "ruby", "go", "golang", "rust", "swift", "kotlin", "scala",
            "r", "matlab", "perl", "php", "bash", "shell", "powershell",
            # Web
            "html", "css", "react", "angular", "vue", "node.js", "nodejs",
            "express", "django", "flask", "fastapi", "spring", "laravel",
            "bootstrap", "tailwind", "jquery", "next.js", "nuxt",
            # Data / ML
            "machine learning", "deep learning", "nlp", "computer vision",
            "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
            "numpy", "matplotlib", "seaborn", "opencv", "huggingface",
            "transformers", "bert", "gpt", "llm", "rag",
            # Data engineering
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
            "cassandra", "dynamodb", "sqlite", "oracle", "spark", "hadoop",
            "kafka", "airflow", "dbt", "etl",
            # Cloud / DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "ansible", "jenkins", "github actions", "ci/cd", "linux",
            "git", "github", "gitlab", "bitbucket",
            # Soft / general
            "agile", "scrum", "jira", "confluence", "rest api", "graphql",
            "microservices", "object-oriented", "oop", "data structures",
            "algorithms", "problem solving", "communication", "leadership",
        ]
        text_lower = text.lower()
        return [skill for skill in KNOWN_SKILLS if skill in text_lower]

    def _extract_section(self, lines: list, headers: list) -> list:
        """Extract lines belonging to a named section."""
        result = []
        in_section = False
        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()
            if any(lower.startswith(h) for h in headers):
                in_section = True
                continue
            if in_section:
                if any(lower.startswith(h) for h in self.SECTION_HEADERS) and lower not in headers:
                    in_section = False
                elif stripped:
                    result.append(stripped)
        return result

    def _detect_sections(self, text: str) -> list:
        text_lower = text.lower()
        return [h for h in self.SECTION_HEADERS if h in text_lower]
