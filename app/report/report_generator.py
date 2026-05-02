"""
app/report/report_generator.py
Saves the ATS analysis result to a JSON report file.
"""

import json
import os
from datetime import datetime


class ReportGenerator:
    """Serializes the pipeline result dict to a formatted JSON file."""

    def save(self, result: dict, output_path: str) -> None:
        result["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  [REPORT] Saved → {output_path}")

    def load(self, report_path: str) -> dict:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
