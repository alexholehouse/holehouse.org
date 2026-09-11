"""Build the V3 note set from the V1 Evernote exports."""

from __future__ import annotations

import os
import sys

from convert import convert

V1 = "/Users/alex/Dropbox/old/mlclass/V1"
V3 = "/Users/alex/Dropbox/old/mlclass/V3"
EXTRA = os.path.dirname(os.path.abspath(__file__)) + "/src_extra"

# (file stem, index label) in reading order.
CHAPTERS = [
    ("01_02_Introduction_regression_analysis_and_gr", "01 and 02: Introduction, Regression Analysis and Gradient Descent"),
    ("03_Linear_algebra_review", "03: Linear Algebra - review"),
    ("04_Linear_Regression_with_multiple_variables", "04: Linear Regression with Multiple Variables"),
    ("05_Octave", "05: Octave [incomplete]"),
    ("06_Logistic_Regression", "06: Logistic Regression"),
    ("07_Regularization", "07: Regularization"),
    ("08_Neural_Networks_Representation", "08: Neural Networks - Representation"),
    ("09_Neural_Networks_Learning", "09: Neural Networks - Learning"),
    ("10_Advice_for_applying_machine_learning", "10: Advice for applying machine learning techniques"),
    ("11_Machine_Learning_System_Design", "11: Machine Learning System Design"),
    ("12_Support_Vector_Machines", "12: Support Vector Machines"),
    ("13_Clustering", "13: Clustering"),
    ("14_Dimensionality_Reduction", "14: Dimensionality Reduction"),
    ("15_Anomaly_Detection", "15: Anomaly Detection"),
    ("16_Recommender_Systems", "16: Recommender Systems"),
    ("17_Large_Scale_Machine_Learning", "17: Large Scale Machine Learning"),
    ("18_Application_Example_OCR", "18: Application Example - Photo OCR"),
    ("19_Course_Summary", "19: Course Summary"),
]

SHORT = {
    "01_02_Introduction_regression_analysis_and_gr": "01/02: Introduction",
    "03_Linear_algebra_review": "03: Linear algebra",
    "04_Linear_Regression_with_multiple_variables": "04: Multivariate regression",
    "05_Octave": "05: Octave",
    "06_Logistic_Regression": "06: Logistic regression",
    "07_Regularization": "07: Regularization",
    "08_Neural_Networks_Representation": "08: Neural networks I",
    "09_Neural_Networks_Learning": "09: Neural networks II",
    "10_Advice_for_applying_machine_learning": "10: Applying ML",
    "11_Machine_Learning_System_Design": "11: System design",
    "12_Support_Vector_Machines": "12: SVMs",
    "13_Clustering": "13: Clustering",
    "14_Dimensionality_Reduction": "14: Dimensionality reduction",
    "15_Anomaly_Detection": "15: Anomaly detection",
    "16_Recommender_Systems": "16: Recommender systems",
    "17_Large_Scale_Machine_Learning": "17: Large scale ML",
    "18_Application_Example_OCR": "18: Photo OCR",
    "19_Course_Summary": "19: Course summary",
}


def source(stem: str) -> str:
    local = os.path.join(V1, stem + ".html")
    return local if os.path.exists(local) else os.path.join(EXTRA, stem + ".html")


def main() -> int:
    os.makedirs(V3, exist_ok=True)
    all_toc = {}

    for i, (stem, label) in enumerate(CHAPTERS):
        prev_stem = CHAPTERS[i - 1][0] if i > 0 else None
        next_stem = CHAPTERS[i + 1][0] if i < len(CHAPTERS) - 1 else None
        meta = {
            "title": label,
            "prev_href": f"{prev_stem}.html" if prev_stem else None,
            "prev_label": SHORT.get(prev_stem, "Previous") if prev_stem else "",
            "next_href": f"{next_stem}.html" if next_stem else None,
            "next_label": SHORT.get(next_stem, "Next") if next_stem else "",
        }
        page, toc = convert(source(stem), meta)
        with open(os.path.join(V3, stem + ".html"), "w", encoding="utf-8") as fh:
            fh.write(page)
        all_toc[stem] = toc
        print(f"{stem:52s} {len(toc):2d} sections  {len(page):>8,d} bytes")

    # Content corrections are applied on top of the mechanical conversion.
    import fixes
    return fixes.main()


if __name__ == "__main__":
    sys.exit(main())
