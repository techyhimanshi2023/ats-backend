"""
app/scorer/ats_scorer.py
Combines all sub-scores into a final ATS score and generates suggestions.
"""


class ATSScorer:
    """
    Weighted scoring model:
      - Keyword match   : 35%
      - Skills match    : 30%
      - Experience      : 20%
      - Format/sections : 15%
    """

    WEIGHTS = {
        "keyword": 0.35,
        "skills":  0.30,
        "experience": 0.20,
        "format": 0.15,
    }

    # Section names that improve ATS parsing
    GOOD_SECTIONS = [
        "summary", "experience", "education", "skills",
        "projects", "certifications", "achievements"
    ]

    def __init__(self):
        pass

    def score(
        self,
        keyword_result: dict,
        skills_result: dict,
        experience_result: dict,
        resume_data: dict,
    ) -> dict:
        """
        Calculate sub-scores and final ATS score.

        Returns:
            dict with ats_score, keyword_match_score, skills_score,
                 experience_score, format_score, suggestions
        """
        keyword_score = keyword_result.get("match_pct", 0.0)
        skills_score  = skills_result.get("match_pct", 0.0)
        experience_score = experience_result.get("experience_score_pct", 0.0)
        format_score  = self._format_score(resume_data)

        ats_score = (
            keyword_score    * self.WEIGHTS["keyword"] +
            skills_score     * self.WEIGHTS["skills"] +
            experience_score * self.WEIGHTS["experience"] +
            format_score     * self.WEIGHTS["format"]
        )

        suggestions = self._generate_suggestions(
            keyword_result, skills_result, experience_result,
            resume_data, format_score
        )

        return {
            "ats_score": round(ats_score, 1),
            "keyword_match_score": round(keyword_score, 1),
            "skills_score": round(skills_score, 1),
            "experience_score": round(experience_score, 1),
            "format_score": round(format_score, 1),
            "suggestions": suggestions,
        }

    # ------------------------------------------------------------------

    def _format_score(self, resume_data: dict) -> float:
        """
        Score based on presence of key sections and basic formatting checks.
        Max 100 pts.
        """
        sections_found = set(resume_data.get("sections_found", []))
        pts = 0.0

        # Section presence (10 pts each, max 70)
        for section in self.GOOD_SECTIONS:
            if section in sections_found:
                pts += 10

        # Contact info (15 pts)
        if resume_data.get("email"):
            pts += 8
        if resume_data.get("phone"):
            pts += 7

        # Name detected (5 pts)
        if resume_data.get("name"):
            pts += 5

        # Raw text not empty (10 pts)
        if len(resume_data.get("raw_text", "")) > 200:
            pts += 10

        return min(pts, 100.0)

    def _generate_suggestions(
        self,
        keyword_result: dict,
        skills_result: dict,
        experience_result: dict,
        resume_data: dict,
        format_score: float,
    ) -> list:
        suggestions = []

        # Keyword suggestions
        missing_kw = keyword_result.get("missing", [])
        if missing_kw:
            top = missing_kw[:5]
            suggestions.append(
                f"Add these missing keywords to your resume: {', '.join(top)}"
            )

        if keyword_result.get("match_pct", 0) < 50:
            suggestions.append(
                "Your keyword match is below 50%. Mirror the job description language more closely."
            )

        # Skills suggestions
        missing_skills = skills_result.get("missing", [])
        if missing_skills:
            top = missing_skills[:5]
            suggestions.append(
                f"The following required skills are missing: {', '.join(top)}. "
                "Add them if you have experience with them."
            )

        # Experience suggestions
        if not experience_result.get("meets_experience_requirement", True):
            yr = experience_result.get("years_required_by_job", 0)
            found = experience_result.get("years_found_in_resume", 0)
            suggestions.append(
                f"Job requires {yr} years of experience; resume shows ~{found}. "
                "Quantify your work history with clear date ranges."
            )

        # Format suggestions
        sections_found = set(resume_data.get("sections_found", []))

        if "summary" not in sections_found:
            suggestions.append(
                "Add a professional Summary or Objective section at the top."
            )
        if "certifications" not in sections_found:
            suggestions.append(
                "Consider adding a Certifications section for relevant credentials."
            )
        if not resume_data.get("email"):
            suggestions.append("Make sure your email address is clearly visible.")
        if not resume_data.get("phone"):
            suggestions.append("Include your phone number in the contact section.")

        if format_score < 60:
            suggestions.append(
                "Improve resume structure: include clearly labeled sections "
                "(Experience, Education, Skills, Projects)."
            )

        return suggestions
