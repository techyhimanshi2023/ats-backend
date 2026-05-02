"""
app/analyzer/experience_analyzer.py
Evaluates resume experience against job requirements.
"""

import re


class ExperienceAnalyzer:
    """
    Compares:
      - Years of experience (resume vs required)
      - Education level match
      - Presence of key sections (summary, projects, certifications)
    """

    DEGREE_RANK = {
        "phd": 5, "ph.d": 5, "doctorate": 5,
        "master": 4, "m.s": 4, "m.tech": 4, "mba": 4,
        "bachelor": 3, "b.s": 3, "b.tech": 3, "b.e": 3,
        "diploma": 2, "associate": 2,
        "high school": 1,
    }

    def __init__(self):
        pass

    def analyze(self, resume_data: dict, job_data: dict) -> dict:
        """
        Args:
            resume_data : dict from ResumeParser
            job_data    : dict from JobParser

        Returns:
            dict with years_found, years_required, education_match,
                 sections_present, experience_score_pct
        """
        resume_text = resume_data.get("clean_text", resume_data.get("raw_text", ""))
        years_in_resume = self._extract_years(resume_text)
        years_required = job_data.get("experience_years", 0)

        edu_score = self._education_match(
            resume_text,
            job_data.get("education_level", "")
        )

        key_sections = ["summary", "projects", "certifications", "achievements"]
        sections_found = resume_data.get("sections_found", [])
        present_sections = [s for s in key_sections if s in sections_found]

        # Score: 0-100
        score = self._compute_score(
            years_in_resume, years_required, edu_score, present_sections
        )

        return {
            "years_found_in_resume": years_in_resume,
            "years_required_by_job": years_required,
            "meets_experience_requirement": years_in_resume >= years_required if years_required else True,
            "education_match_score": edu_score,
            "key_sections_present": present_sections,
            "experience_score_pct": score,
        }

    # ------------------------------------------------------------------

    def _extract_years(self, text: str) -> int:
        """
        Best-guess total years of experience from resume text.
        Looks for year ranges like 2019–2023 and sums durations,
        or explicit statements like '5 years of experience'.
        """
        # Explicit statements
        explicit = re.findall(
            r"(\d+)\s*\+?\s*years?\s+of\s+(?:professional\s+)?experience",
            text, re.IGNORECASE
        )
        if explicit:
            return max(int(x) for x in explicit)

        # Year ranges like 2018 - 2022 or Jan 2018 – Dec 2022
        ranges = re.findall(r"\b(20\d{2})\s*[-–]\s*(20\d{2}|present|current)\b",
                            text, re.IGNORECASE)
        import datetime
        current_year = datetime.datetime.now().year
        total = 0
        for start, end in ranges:
            start_yr = int(start)
            end_yr = current_year if end.lower() in ("present", "current") else int(end)
            total += max(0, end_yr - start_yr)

        return min(total, 30)  # cap at 30 to avoid bogus sums

    def _education_match(self, resume_text: str, required_level: str) -> float:
        """
        Returns 0.0–1.0 indicating how well resume education meets requirement.
        """
        if not required_level:
            return 1.0

        required_rank = self.DEGREE_RANK.get(required_level.lower(), 0)
        resume_lower = resume_text.lower()

        resume_rank = 0
        for degree, rank in self.DEGREE_RANK.items():
            if degree in resume_lower:
                resume_rank = max(resume_rank, rank)

        if resume_rank >= required_rank:
            return 1.0
        elif resume_rank == required_rank - 1:
            return 0.7
        elif resume_rank > 0:
            return 0.4
        return 0.0

    def _compute_score(
        self,
        years_found: int,
        years_required: int,
        edu_score: float,
        present_sections: list
    ) -> float:
        # Experience years (50 pts)
        if years_required == 0:
            exp_pts = 50.0
        elif years_found >= years_required:
            exp_pts = 50.0
        else:
            exp_pts = (years_found / years_required) * 50

        # Education (30 pts)
        edu_pts = edu_score * 30

        # Sections (20 pts, up to 4 sections × 5 pts each)
        section_pts = min(len(present_sections) * 5, 20)

        return round(exp_pts + edu_pts + section_pts, 1)
