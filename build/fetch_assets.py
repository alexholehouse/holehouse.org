"""Collect every image referenced by the V1 chapters into V3.

Local copies are preferred; anything absent locally is pulled from the published
copy of the notes at holehouse.org. Output directory names are normalised to
exactly match the ``src`` attributes so the tree works on a case-sensitive server.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request

V1 = "/Users/alex/Dropbox/old/mlclass/V1"
V3 = "/Users/alex/Dropbox/old/mlclass/V3"
EXTRA = os.path.dirname(os.path.abspath(__file__)) + "/src_extra"
BASE = "https://www.holehouse.org/mlclass/"

CHAPTERS = [
    "01_02_Introduction_regression_analysis_and_gr",
    "03_Linear_algebra_review",
    "04_Linear_Regression_with_multiple_variables",
    "05_Octave",
    "06_Logistic_Regression",
    "07_Regularization",
    "08_Neural_Networks_Representation",
    "09_Neural_Networks_Learning",
    "10_Advice_for_applying_machine_learning",
    "11_Machine_Learning_System_Design",
    "12_Support_Vector_Machines",
    "13_Clustering",
    "14_Dimensionality_Reduction",
    "15_Anomaly_Detection",
    "16_Recommender_Systems",
    "17_Large_Scale_Machine_Learning",
    "18_Application_Example_OCR",
    "19_Course_Summary",
]


def source_path(stem: str) -> str:
    """Return the on-disk source for a chapter, falling back to the fetched copy."""
    local = os.path.join(V1, stem + ".html")
    return local if os.path.exists(local) else os.path.join(EXTRA, stem + ".html")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 mlclass-v3-rebuild"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    copied = fetched = failed = 0
    failures: list[str] = []

    for stem in CHAPTERS:
        src_file = source_path(stem)
        html = open(src_file, encoding="utf-8", errors="replace").read()
        srcs = re.findall(r'<img[^>]*\ssrc="([^"]+)"', html)
        # Preserve order but drop duplicates.
        seen: set[str] = set()
        srcs = [s for s in srcs if not (s in seen or seen.add(s))]

        for rel in srcs:
            out = os.path.join(V3, rel)
            if os.path.exists(out):
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)

            local = os.path.join(V1, rel)
            # macOS is case-insensitive, so this also finds the lowercased dirs.
            if os.path.exists(local) and os.path.getsize(local) > 0:
                shutil.copyfile(local, out)
                copied += 1
                continue

            url = BASE + urllib.parse.quote(rel)
            try:
                data = fetch(url)
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                failures.append(f"{rel}: {exc}")
                continue
            if not data:
                failed += 1
                failures.append(f"{rel}: empty response")
                continue
            with open(out, "wb") as fh:
                fh.write(data)
            fetched += 1
            time.sleep(0.15)

        print(f"{stem:52s} done", flush=True)

    print(f"\ncopied={copied} fetched={fetched} failed={failed}")
    for line in failures:
        print("  FAIL", line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
