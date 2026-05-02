"""
app/analyzer/skills_analyzer.py
Compares resume skills against job-required skills.
"""


class SkillsAnalyzer:
    """
    Calculates how many of the job's required skills appear in the resume.
    Produces:
      - matched       : skills present in both
      - missing       : required skills absent from resume
      - match_pct     : percentage of required skills matched
      - bonus_skills  : resume skills not required (extras, positive)
    """

    def __init__(self):
        # Skill aliases / synonyms  (canonical → [aliases])
        self.ALIASES = {
            "node.js": ["nodejs", "node js"],
            "machine learning": ["ml"],
            "deep learning": ["dl"],
            "natural language processing": ["nlp"],
            "object-oriented": ["oop", "object oriented"],
            "postgresql": ["postgres"],
            "kubernetes": ["k8s"],
            "ci/cd": ["continuous integration", "continuous delivery", "continuous deployment"],
            "rest api": ["restful", "rest"],
            "github actions": ["gh actions"],
        }

    def analyze(self, resume_skills: list, job_skills: list) -> dict:
        """
        Args:
            resume_skills : skills extracted from resume (list of strings)
            job_skills    : required skills from job description

        Returns:
            dict with matched, missing, match_pct, bonus_skills
        """
        resume_set = self._normalize_set(resume_skills)
        job_set = self._normalize_set(job_skills)

        matched = []
        missing = []

        for skill in job_skills:
            norm = skill.lower().strip()
            if self._is_present(norm, resume_set):
                matched.append(skill)
            else:
                missing.append(skill)

        total = len(job_skills)
        match_pct = (len(matched) / total * 100) if total > 0 else 0.0

        # Bonus: resume skills not explicitly required
        bonus = [s for s in resume_skills if s.lower().strip() not in job_set]

        return {
            "matched": matched,
            "missing": missing,
            "total_required": total,
            "match_pct": round(match_pct, 2),
            "bonus_skills": bonus,
        }

    # ------------------------------------------------------------------

    def _normalize_set(self, skills: list) -> set:
        return {s.lower().strip() for s in skills}

    def _is_present(self, skill: str, resume_set: set) -> bool:
        if skill in resume_set:
            return True
        # Check aliases
        for canonical, aliases in self.ALIASES.items():
            if skill == canonical or skill in aliases:
                # Check if canonical or any alias is in resume
                if canonical in resume_set:
                    return True
                if any(a in resume_set for a in aliases):
                    return True
        return False
