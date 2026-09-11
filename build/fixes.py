"""Apply the typo and wording corrections to the generated V3 pages.

Kept separate from convert.py on purpose: the conversion is mechanical and
reversible, whereas these change the words on the page and so need to be
reviewable one by one. Every entry is (chapter, before, after, count) and the
run fails loudly if a replacement does not match exactly the expected number of
times, so a fix can never apply silently or to the wrong place.
"""

from __future__ import annotations

import os
import sys

V3 = "/Users/alex/Dropbox/old/mlclass/V3"

# (file stem, before, after, expected occurrences, note for the changelog)
FIXES: list[tuple[str, str, str, int, str]] = [
    # -- 01 and 02 -----------------------------------------------------------
    ("01_02_Introduction_regression_analysis_and_gr",
     "understand cotour plots", "understand contour plots", 1, "typo: cotour -> contour"),
    ("01_02_Introduction_regression_analysis_and_gr",
     "center of the countour plot", "center of the contour plot", 1, "typo: countour -> contour"),
    ("01_02_Introduction_regression_analysis_and_gr",
     "Do the following until covergence", "Do the following until convergence", 1,
     "typo: covergence -> convergence"),
    ("01_02_Introduction_regression_analysis_and_gr",
     "Two key functins", "Two key functions", 1, "typo: functins -> functions"),
    ("01_02_Introduction_regression_analysis_and_gr",
     "towards the mimum (down) will greate a negative derivative",
     "towards the minimum (down) will create a negative derivative", 1,
     "typos: mimum -> minimum, greate -> create"),
    ("01_02_Introduction_regression_analysis_and_gr",
     "complex hyothesis with two pariables", "complex hypothesis with two variables", 1,
     "typos: hyothesis -> hypothesis, pariables -> variables"),
    ("01_02_Introduction_regression_analysis_and_gr",
     "linear regression modles", "linear regression models", 1, "typo: modles -> models"),
    ("01_02_Introduction_regression_analysis_and_gr",
     "because of the summartion term", "because of the summation term", 1,
     "typo: summartion -> summation"),
    ("01_02_Introduction_regression_analysis_and_gr",
     "turn records in knowledges", "turn records into knowledge", 1,
     "grammar: 'turn records in knowledges' -> 'turn records into knowledge'"),
    ("01_02_Introduction_regression_analysis_and_gr",
     "Samuels wrote a checkers playing program", "Samuel wrote a checkers playing program", 1,
     "name: Arthur Samuel, not Samuels"),

    # -- 04 ------------------------------------------------------------------
    ("04_Linear_Regression_with_multiple_variables",
     "partial derivative of of the", "partial derivative of the", 1, "doubled word: 'of of'"),
    ("04_Linear_Regression_with_multiple_variables",
     "Very hard to tel in advance", "Very hard to tell in advance", 1, "typo: tel -> tell"),

    # -- 06 ------------------------------------------------------------------
    ("06_Logistic_Regression",
     "and thats exactly correct", "and that's exactly correct", 1, "typo: thats -> that's"),
    ("06_Logistic_Regression",
     "[jval, gradent] = costFunction", "[jval, gradient] = costFunction", 1,
     "code typo: gradent -> gradient"),
    ("06_Logistic_Regression",
     "[optTheta, funtionVal, exitFlag]", "[optTheta, functionVal, exitFlag]", 1,
     "code typo: funtionVal -> functionVal"),
    ("06_Logistic_Regression",
     "options= optimset", "options = optimset", 1, "code spacing around ="),
    ("06_Logistic_Regression",
     "initialTheta= zeros(2,1)", "initialTheta = zeros(2,1)", 1, "code spacing around ="),
    ("06_Logistic_Regression",
     "exitFlag]= fminunc", "exitFlag] = fminunc", 1, "code spacing around ="),

    # -- 07 ------------------------------------------------------------------
    ("07_Regularization",
     "doesn't extend to to the lambda term", "doesn't extend to the lambda term", 1,
     "doubled word: 'to to'"),

    # -- 08 ------------------------------------------------------------------
    ("08_Neural_Networks_Representation",
     "is a 3x1 vecor", "is a 3x1 vector", 1, "typo: vecor -> vector"),

    # -- 09 ------------------------------------------------------------------
    ("09_Neural_Networks_Learning",
     "the Backproc implementation is correc", "the Backprop implementation is correct", 1,
     "typos: Backproc -> Backprop, correc -> correct"),

    # -- 11 ------------------------------------------------------------------
    ("11_Machine_Learning_System_Design",
     "we have have a higher recall", "we have a higher recall", 1, "doubled word: 'have have'"),

    # -- 12 ------------------------------------------------------------------
    ("12_Support_Vector_Machines",
     "power is using diferent kernels", "power is using different kernels", 1,
     "typo: diferent -> different"),
    ("12_Support_Vector_Machines",
     "Disussed more later", "Discussed more later", 1, "typo: Disussed -> Discussed"),
    ("12_Support_Vector_Machines",
     "Gaussian Kernel this evalues to 1", "Gaussian Kernel this evaluates to 1", 1,
     "typo: evalues -> evaluates"),
    ("12_Support_Vector_Machines",
     "why the SVM choses this hypothesis", "why the SVM chooses this hypothesis", 1,
     "typo: choses -> chooses"),
    ("12_Support_Vector_Machines",
     "by contrast is the the chosen by the SVM", "by contrast is the one chosen by the SVM", 1,
     "dropped word: 'is the the chosen' -> 'is the one chosen'"),
    ("12_Support_Vector_Machines",
     "on to to the", "on to the", 1, "doubled word: 'to to'"),

    # -- 13 ------------------------------------------------------------------
    ("13_Clustering",
     "the index of the closes variable of cluster centroid closes to",
     "the index of the closest variable of cluster centroid closest to", 1,
     "typo: closes -> closest (twice in one line)"),
    ("13_Clustering",
     "that value is one the the clusters", "that value is one of the clusters", 1,
     "dropped word: 'one the the clusters' -> 'one of the clusters'"),
    ("13_Clustering",
     "you don't get a a nice line", "you don't get a nice line", 1, "doubled word: 'a a'"),

    # -- 15 ------------------------------------------------------------------
    ("15_Anomaly_Detection",
     "the mean (n-dimenisonal vector)", "the mean (n-dimensional vector)", 1,
     "typo: dimenisonal -> dimensional"),
    ("15_Anomaly_Detection",
     "if you think something iss anomalous", "if you think something is anomalous", 1,
     "typo: iss -> is"),
    ("15_Anomaly_Detection",
     "concentric circles around the the means", "concentric circles around the means", 1,
     "doubled word: 'the the'"),

    # -- 17 ------------------------------------------------------------------
    ("17_Large_Scale_Machine_Learning",
     "greater number of entires per average", "greater number of entries per average", 1,
     "typo: entires -> entries"),
    ("17_Large_Scale_Machine_Learning",
     "Send to to a centralized master server", "Send to a centralized master server", 1,
     "doubled word: 'to to'"),

    # -- 18 ------------------------------------------------------------------
    ("18_Application_Example_OCR",
     "makes it a bit easer", "makes it a bit easier", 1, "typo: easer -> easier"),
]


def main() -> int:
    by_file: dict[str, list] = {}
    for stem, before, after, count, note in FIXES:
        by_file.setdefault(stem, []).append((before, after, count, note))

    failures: list[str] = []
    applied = 0

    for stem, items in by_file.items():
        path = os.path.join(V3, stem + ".html")
        text = open(path, encoding="utf-8").read()
        for before, after, count, note in items:
            found = text.count(before)
            if found != count:
                failures.append(f"{stem}: expected {count}x {before!r}, found {found}")
                continue
            text = text.replace(before, after)
            applied += 1
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    print(f"applied {applied}/{len(FIXES)} fixes across {len(by_file)} chapters")
    for line in failures:
        print("  FAILED:", line)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
