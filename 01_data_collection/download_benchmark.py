#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 01_data_collection/download_benchmark.py
=============================================================================
 Downloads the cognitive-bias benchmark of Malberg, Poletukhin, Schuster &
 Groh (2025), "A Comprehensive Evaluation of Cognitive Biases in LLMs"
 (arXiv:2410.15413, NLP4DH 2025) from HuggingFace:

     https://huggingface.co/datasets/tum-nlp/cognitive-biases-in-llms

 The dataset contains 30,000 tests (30 biases x 200 scenarios x 5 instances).
 Each row has: bias, scenario, control, treatment, metric_params.

 The script caches the raw data locally, prints per-bias item counts, and
 verifies that every bias targeted in config.yaml exists in the dataset
 (failing loudly with the full list of valid names if not).

 Usage:
   python 01_data_collection/download_benchmark.py
   python 01_data_collection/download_benchmark.py --force   # re-download

 Saves:
   01_data_collection/outputs/benchmark_raw.json   (list of row dicts)
   01_data_collection/outputs/download_log.txt     (counts + validation log)
=============================================================================
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HF_REPO = "tum-nlp/cognitive-biases-in-llms"
HF_FILE = "cognitive-biases-in-llms.csv"

LOG_LINES = []


def log(line: str = "") -> None:
    print(line, flush=True)
    LOG_LINES.append(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the cognitive-bias benchmark")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the local cache exists")
    args = parser.parse_args()

    import yaml
    import pandas as pd
    from huggingface_hub import hf_hub_download

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    out_path = ROOT / cfg["paths"]["benchmark_raw"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(" ANCHORED MINDS -- BENCHMARK DOWNLOAD")
    log(f" Source    : hf.co/datasets/{HF_REPO}")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log("=" * 70)

    # ---------------------------------------------------------------- fetch --
    if out_path.exists() and not args.force:
        log(f"\n[1/3] Local cache found at {out_path.relative_to(ROOT)}")
        log("      Skipping download (use --force to re-download).")
        df = pd.DataFrame(json.loads(out_path.read_text()))
    else:
        log(f"\n[1/3] Downloading {HF_FILE} from HuggingFace ...")
        csv_path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
        log(f"      Downloaded to HF cache: {csv_path}")
        df = pd.read_csv(csv_path)
        log(f"      Loaded {len(df):,} rows, columns: {list(df.columns)}")
        out_path.write_text(json.dumps(df.to_dict(orient="records")),
                            encoding="utf-8")
        log(f"      Cached -> {out_path.relative_to(ROOT)} "
            f"({out_path.stat().st_size / 1e6:.1f} MB)")

    # ------------------------------------------------------------- summarize --
    log(f"\n[2/3] Dataset summary")
    log(f"      Total tests : {len(df):,}")
    log(f"      Biases      : {df['bias'].nunique()}")
    log(f"      Scenarios   : {df['scenario'].nunique()}")
    log("\n      Items per bias:")
    counts = df["bias"].value_counts().sort_index()
    for bias, n in counts.items():
        marker = "  <-- TARGET" if bias in cfg["biases"] else ""
        log(f"        {bias:<34}{n:>6,}{marker}")

    # ---------------------------------------------------------------- verify --
    log(f"\n[3/3] Validating config.yaml target biases against dataset")
    available = set(df["bias"].unique())
    missing = [b for b in cfg["biases"] if b not in available]
    for b in cfg["biases"]:
        mark = "\u2713" if b in available else "\u2717"
        log(f"      [{mark}] {b}")

    ok = not missing
    log("\n" + "=" * 70)
    if ok:
        log(" RESULT: DOWNLOAD OK -- all target biases found \u2713")
        log(" Next step: python 01_data_collection/build_test_battery.py")
    else:
        log(f" RESULT: FAILED \u2717 -- {len(missing)} target bias name(s) not in "
            f"dataset: {missing}")
        log(" Fix the 'biases' list in config.yaml using the exact names above.")
    log("=" * 70)

    log_path = Path(__file__).resolve().parent / "outputs" / "download_log.txt"
    log_path.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    print(f"\nLog saved -> {log_path}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
