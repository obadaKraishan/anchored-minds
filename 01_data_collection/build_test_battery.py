#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 01_data_collection/build_test_battery.py
=============================================================================
 Builds and FREEZES the stimulus battery from the raw benchmark.

 For each target bias in config.yaml, samples n_items_per_bias test items
 (seeded, reproducible), parses the answer-option scale out of the prompt
 text, and assigns stable item IDs. The resulting battery is the study's
 informal preregistration: after this file is written, stimuli never change.

 The script REFUSES to overwrite an existing battery unless --overwrite is
 passed, protecting the freeze.

 Each battery item carries:
   item_id          e.g. ANCH-003
   bias             benchmark bias label
   scenario         decision-making scenario string
   control_text     full control prompt (no bias manipulation)
   treatment_text   full treatment prompt (with bias manipulation)
   n_options        number of answer options on the ordinal scale
   option_labels    list of the option label strings
   k, flip_treatment  direction parameters from the benchmark's metric_params

 Usage:
   python 01_data_collection/build_test_battery.py
   python 01_data_collection/build_test_battery.py --overwrite

 Saves (in 01_data_collection/outputs/battery/):
   final_battery.csv / final_battery.json   (the frozen stimulus set)
   battery_summary.txt                      (selection log for the Method section)
=============================================================================
"""

import argparse
import ast
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_LINES = []


def log(line: str = "") -> None:
    print(line, flush=True)
    LOG_LINES.append(line)


OPTION_RE = re.compile(r"^Option (\d+):\s*(.+)\s*$", re.MULTILINE)


def parse_options(prompt_text: str):
    """Extract the ordinal answer options from a benchmark prompt."""
    matches = OPTION_RE.findall(prompt_text)
    return [label for _, label in matches]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and freeze the test battery")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow overwriting an existing frozen battery")
    args = parser.parse_args()

    import yaml
    import pandas as pd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    seed = cfg["project"]["seed"]
    n_items = cfg["experiment"]["n_items_per_bias"]
    battery_dir = ROOT / cfg["paths"]["battery_dir"]
    battery_dir.mkdir(parents=True, exist_ok=True)
    csv_path = battery_dir / "final_battery.csv"
    json_path = battery_dir / "final_battery.json"

    log("=" * 70)
    log(" ANCHORED MINDS -- BUILD TEST BATTERY")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log(f" Seed      : {seed} | items per bias: {n_items}")
    log("=" * 70)

    # ------------------------------------------------------------ freeze guard
    if csv_path.exists() and not args.overwrite:
        log("\n[ABORT] A frozen battery already exists at "
            f"{csv_path.relative_to(ROOT)}.")
        log("        The battery is the study's preregistration -- it should "
            "not change after Day 1.")
        log("        If you REALLY need to rebuild it, rerun with --overwrite.")
        return 1

    # ------------------------------------------------------------------- load
    raw_path = ROOT / cfg["paths"]["benchmark_raw"]
    if not raw_path.exists():
        log(f"\n[ERROR] Raw benchmark not found at {raw_path.relative_to(ROOT)}.")
        log("        Run: python 01_data_collection/download_benchmark.py")
        return 1
    df = pd.DataFrame(json.loads(raw_path.read_text()))
    log(f"\n[1/4] Loaded raw benchmark: {len(df):,} rows, "
        f"{df['bias'].nunique()} biases")

    # ---------------------------------------------------------------- validate
    available = set(df["bias"].unique())
    missing = [b for b in cfg["biases"] if b not in available]
    if missing:
        log(f"\n[ERROR] Target bias name(s) not in dataset: {missing}")
        log(f"        Valid names: {sorted(available)}")
        return 1
    log(f"[2/4] All {len(cfg['biases'])} target biases present \u2713")

    # ------------------------------------------------------------------ sample
    rng = random.Random(seed)
    items = []
    log(f"\n[3/4] Sampling {n_items} items per bias (seeded, "
        "distinct scenarios preferred)")
    for bias in cfg["biases"]:
        sub = df[df["bias"] == bias].reset_index(drop=True)
        # prefer scenario diversity: sample distinct scenarios first
        scenarios = sorted(sub["scenario"].unique())
        rng.shuffle(scenarios)
        chosen_rows = []
        for sc in scenarios[:n_items]:
            pool = sub[sub["scenario"] == sc]
            chosen_rows.append(pool.iloc[rng.randrange(len(pool))])
        prefix = re.sub(r"[^A-Z]", "", bias.upper())[:4] or "BIAS"
        n_parse_fail = 0
        for i, row in enumerate(chosen_rows, start=1):
            ctrl_opts = parse_options(row["control"])
            trt_opts = parse_options(row["treatment"])
            if not ctrl_opts or len(ctrl_opts) != len(trt_opts):
                n_parse_fail += 1
                continue
            mp = ast.literal_eval(row["metric_params"]) \
                if isinstance(row["metric_params"], str) else row["metric_params"]
            items.append({
                "item_id": f"{prefix}-{i:03d}",
                "bias": bias,
                "scenario": row["scenario"],
                "control_text": row["control"],
                "treatment_text": row["treatment"],
                "n_options": len(ctrl_opts),
                "option_labels": ctrl_opts,
                "k": int(mp.get("k", 1)),
                "flip_treatment": bool(mp.get("flip_treatment", False)),
            })
        n_ok = sum(1 for it in items if it["bias"] == bias)
        log(f"      {bias:<28} sampled {n_ok}/{n_items}"
            + (f"  ({n_parse_fail} dropped: option-parse failure)"
               if n_parse_fail else ""))

    # -------------------------------------------------------------------- save
    bat = pd.DataFrame(items)
    bat_json = bat.copy()
    bat.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    log(f"\n[4/4] Battery FROZEN: {len(bat)} items")
    log(f"      CSV  -> {csv_path.relative_to(ROOT)}")
    log(f"      JSON -> {json_path.relative_to(ROOT)}")

    # option-scale distribution (useful for the Method section)
    log("\n      Option-scale sizes: "
        + ", ".join(f"{n}-point x{c}"
                    for n, c in bat["n_options"].value_counts()
                                                .sort_index().items()))
    log("\n" + "=" * 70)
    log(" RESULT: BATTERY BUILT \u2713   (do not rebuild after experiments start)")
    log(" Next step: python 01_data_collection/run_experiments.py --smoke-test")
    log("=" * 70)

    summary_path = battery_dir / "battery_summary.txt"
    summary_path.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    print(f"\nSummary saved -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
