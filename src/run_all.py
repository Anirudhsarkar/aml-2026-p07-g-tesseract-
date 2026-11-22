"""Run the whole pipeline in order. From the repository root:

    python src/run_all.py                 # everything (needs data/raw/online_retail_II.xlsx)
    python src/run_all.py --from-step 3   # resume from step 3 (RFM)
    python src/run_all.py --skip-report   # skip the report draft
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("Phase 1  cleaning", "clean_retail_data.py"),
    ("Phase 2  EDA", "run_eda.py"),
    ("Phase 3  RFM features", "build_rfm.py"),
    ("Phase 4a K-Means baseline", "clustering.py"),
    ("Phase 4b hierarchical, DBSCAN, stability", "clustering_advanced.py"),
    ("Phase 5  segments and recommendations", "segments.py"),
    ("Phase 6  classification and evaluation", "classification.py"),
    ("Phase 7  report draft", "make_report.py"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-step", type=int, default=1, help="1-based step to start from")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--input", type=Path, help="Workbook path passed to the cleaning step")
    args = parser.parse_args()
    for number, (label, script) in enumerate(STEPS, start=1):
        if number < args.from_step or (args.skip_report and script == "make_report.py"):
            continue
        print(f"\n===== Step {number}: {label} ({script}) =====", flush=True)
        start = time.time()
        command = [sys.executable, str(ROOT / "src" / script)]
        if number == 1 and args.input:
            command.extend(["--input", str(args.input)])
        elif number == 1:
            default_input = ROOT / "data" / "raw" / "online_retail_II.xlsx"
            bundled_input = ROOT / "online+retail+ii" / "online_retail_II.xlsx"
            if not default_input.exists() and bundled_input.exists():
                command.extend(["--input", str(bundled_input)])
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"----- done in {time.time() - start:.1f}s", flush=True)
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
