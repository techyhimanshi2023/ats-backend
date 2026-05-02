"""
app/pipeline.py
Orchestrates the full ATS analysis pipeline.
"""

import os
import json
from app.parser.resume_parser import ResumeParser
from app.parser.job_parser import JobParser
from app.analyzer.keyword_analyzer import KeywordAnalyzer
from app.analyzer.skills_analyzer import SkillsAnalyzer
from app.analyzer.experience_analyzer import ExperienceAnalyzer
from app.scorer.ats_scorer import ATSScorer
from app.report.report_generator import ReportGenerator
from app.utils.text_cleaner import clean_text


class ATSPipeline:
    """
    End-to-end pipeline:
      1. Parse resume & job description
      2. Analyze keywords (exact + BERT semantic)
      3. Analyze skills & experience
      4. Score with weighted model
      5. Predict pass/fail with Logistic Regression (if model trained)
      6. Generate report
    """

    def __init__(self, verbose: bool = False, use_bert: bool = True, use_lr: bool = True):
        self.verbose = verbose
        self.use_lr = use_lr
        self.resume_parser = ResumeParser()
        self.job_parser = JobParser()
        self.keyword_analyzer = KeywordAnalyzer(use_bert=use_bert)
        self.skills_analyzer = SkillsAnalyzer()
        self.experience_analyzer = ExperienceAnalyzer()
        self.scorer = ATSScorer()
        self.report_generator = ReportGenerator()

        # LR predictor — loaded only if model file exists
        self._lr_predictor = None
        if use_lr:
            self._init_lr_predictor()

    def _init_lr_predictor(self):
        try:
            from app.ml.ats_predictor import ATSPredictor
            predictor = ATSPredictor()
            if predictor.load():
                self._lr_predictor = predictor
                self._log("Logistic Regression model loaded.")
            else:
                self._log("No trained LR model found. Run train.py to enable ATS prediction.")
        except Exception as e:
            self._log(f"LR predictor unavailable: {e}")

    def _log(self, message: str):
        if self.verbose:
            print(f"  [PIPELINE] {message}")

    def run(self, resume_path: str, job_description: str, output_path: str = "outputs/ats_report.json") -> dict:
        self._log(f"Parsing resume: {resume_path}")
        resume_data = self.resume_parser.parse(resume_path)
        resume_data["clean_text"] = clean_text(resume_data.get("raw_text", ""))

        self._log("Parsing job description...")
        job_data = self.job_parser.parse(job_description)

        self._log("Analyzing keywords (exact + BERT)...")
        keyword_result = self.keyword_analyzer.analyze(
            resume_text=resume_data["clean_text"],
            job_keywords=job_data["keywords"]
        )

        self._log("Analyzing skills...")
        skills_result = self.skills_analyzer.analyze(
            resume_skills=resume_data.get("skills", []),
            job_skills=job_data.get("required_skills", [])
        )

        self._log("Analyzing experience...")
        experience_result = self.experience_analyzer.analyze(
            resume_data=resume_data,
            job_data=job_data
        )

        self._log("Scoring resume...")
        scores = self.scorer.score(
            keyword_result=keyword_result,
            skills_result=skills_result,
            experience_result=experience_result,
            resume_data=resume_data
        )

        # LR prediction
        lr_prediction = None
        if self._lr_predictor:
            self._log("Running Logistic Regression ATS predictor...")
            try:
                lr_prediction = self._lr_predictor.predict(
                    resume_text=resume_data["clean_text"],
                    job_text=job_data["clean_text"],
                    keyword_match_pct=keyword_result.get("match_pct", 0),
                    skills_match_pct=skills_result.get("match_pct", 0),
                    years_experience=experience_result.get("years_found_in_resume", 0),
                )
            except Exception as e:
                self._log(f"LR prediction failed: {e}")

        result = {
            "resume_file": os.path.basename(resume_path),
            "ats_score": scores["ats_score"],
            "keyword_match_score": scores["keyword_match_score"],
            "skills_score": scores["skills_score"],
            "experience_score": scores["experience_score"],
            "format_score": scores["format_score"],
            "bert_semantic_score": keyword_result.get("semantic_score", 0.0),
            "matched_keywords": keyword_result.get("matched", []),
            "semantic_matched_keywords": keyword_result.get("semantic_matched", []),
            "missing_keywords": keyword_result.get("missing", []),
            "top_missing_keywords": keyword_result.get("missing", [])[:10],
            "matched_skills": skills_result.get("matched", []),
            "missing_skills": skills_result.get("missing", []),
            "experience_details": experience_result,
            "resume_sections_found": resume_data.get("sections_found", []),
            "suggestions": scores.get("suggestions", []),
            "lr_prediction": lr_prediction,
        }

        self._log("Generating report...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.report_generator.save(result, output_path)

        return result
