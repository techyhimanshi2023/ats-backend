"""
app/parser/job_parser.py
Parses a raw job description string into structured data.
"""

import re
from app.utils.text_cleaner import clean_text


class JobParser:
    """
    Extracts structured information from a job description:
      - keywords        : important words/phrases
      - required_skills : explicit skills mentioned
      - preferred_skills: nice-to-have skills
      - experience_years: years of experience requested
      - education_level : degree requirement if mentioned
      - job_title       : inferred job title
    """

    SKILL_KEYWORDS = [
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby",
        "go", "golang", "rust", "swift", "kotlin", "scala", "r", "matlab",
        "php", "bash", "html", "css", "react", "angular", "vue", "node.js",
        "nodejs", "express", "django", "flask", "fastapi", "spring",
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "ci/cd", "git", "github", "linux", "rest api", "graphql",
        "microservices", "agile", "scrum", "spark", "hadoop", "kafka",
        "airflow", "dbt", "etl", "llm", "rag", "transformers", "bert",
        "object-oriented", "oop", "data structures", "algorithms",
    ]

    EDUCATION_LEVELS = {
        "phd": 5, "ph.d": 5, "doctorate": 5,
        "master": 4, "m.s": 4, "m.tech": 4, "mba": 4,
        "bachelor": 3, "b.s": 3, "b.tech": 3, "b.e": 3, "undergraduate": 3,
        "diploma": 2, "associate": 2,
        "high school": 1,
    }

    def __init__(self):
        pass

    def parse(self, job_description: str) -> dict:
        """
        Parse raw job description text.

        Returns:
            dict with keywords, required_skills, preferred_skills,
                 experience_years, education_level, job_title.
        """
        cleaned = clean_text(job_description)
        lines = job_description.splitlines()

        return {
            "raw_text": job_description,
            "clean_text": cleaned,
            "job_title": self._extract_title(lines),
            "keywords": self._extract_keywords(cleaned),
            "required_skills": self._extract_skills(job_description, section="required"),
            "preferred_skills": self._extract_skills(job_description, section="preferred"),
            "experience_years": self._extract_experience_years(job_description),
            "education_level": self._extract_education(job_description),
        }

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_title(self, lines: list) -> str:
        """First non-empty line is usually the job title."""
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 80:
                return line
        return ""

    def _extract_keywords(self, text: str) -> list:
        """
        Combine skill keywords + important nouns found in the JD.
        Returns a deduplicated list, lowercased.
        """
        text_lower = text.lower()
        found = [kw for kw in self.SKILL_KEYWORDS if kw in text_lower]

        # Also add single-word capitalized tokens that appear multiple times
        # as they're likely domain-specific terms
        words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
        freq: dict = {}
        for w in words:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
        important = [w for w, c in freq.items() if c >= 2 and len(w) > 3]

        return list(dict.fromkeys(found + important))

    def _extract_skills(self, text: str, section: str = "required") -> list:
        """
        Try to pull skills specifically from Required / Preferred sections.
        Falls back to scanning the whole text.
        """
        text_lower = text.lower()

        # Determine section boundaries
        if section == "required":
            markers = ["required", "requirements", "must have", "qualifications"]
        else:
            markers = ["preferred", "nice to have", "bonus", "plus", "desired"]

        section_text = ""
        for marker in markers:
            idx = text_lower.find(marker)
            if idx != -1:
                section_text = text[idx: idx + 800]
                break

        search_text = section_text if section_text else text
        search_lower = search_text.lower()
        return [skill for skill in self.SKILL_KEYWORDS if skill in search_lower]

    def _extract_experience_years(self, text: str) -> int:
        """
        Find patterns like '3+ years', '2-4 years', 'five years'.
        Returns the minimum number found, or 0.
        """
        patterns = [
            r"(\d+)\s*\+?\s*years?\s+of\s+experience",
            r"(\d+)\s*[-–]\s*\d+\s*years?",
            r"minimum\s+(\d+)\s+years?",
            r"at\s+least\s+(\d+)\s+years?",
            r"(\d+)\s+years?\s+experience",
        ]
        numbers = []
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                numbers.append(int(m.group(1)))
        return min(numbers) if numbers else 0

    def _extract_education(self, text: str) -> str:
        text_lower = text.lower()
        highest = ""
        highest_level = 0
        for degree, level in self.EDUCATION_LEVELS.items():
            if degree in text_lower and level > highest_level:
                highest = degree
                highest_level = level
        return highest
