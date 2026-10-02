#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 00_setup/cost_estimator.py
=============================================================================
 Estimates total API calls, tokens, and USD cost for the full experimental
 grid BEFORE any money is spent. Run after setup_check.py passes.

 The grid:
   calls(model) = n_biases x n_items_per_bias x n_conditions(2: control,
                  biased) x n_budgets(model) x n_samples_per_cell

   - reasoning models run every budget in config.budgets
   - non-reasoning models run only budget 0
   - thinking tokens are billed as OUTPUT tokens: for budget b > 0 we assume
     ~60% of the ceiling is actually consumed (conservative planning figure,
     reported in the output so you can adjust)

 Usage:
   python 00_setup/cost_estimator.py
   python 00_setup/cost_estimator.py --thinking-utilization 0.8

 Saves:
   00_setup/cost_estimate.csv   (per model x budget breakdown)
   00_setup/cost_estimate.txt   (formatted summary table)
=============================================================================
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_CONDITIONS = 2  # control vs. biased variant of every item


def main() -> int:
    parser = argparse.ArgumentParser(description="Anchored Minds cost estimator")
    parser.add_argument("--thinking-utilization", type=float, default=0.6,
                        help="assumed fraction of the thinking budget actually "
                             "consumed (default 0.6)")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    exp = cfg["experiment"]

    n_biases = len(cfg["biases"])
    n_items = exp["n_items_per_bias"]
    n_samples = exp["n_samples_per_cell"]
    budgets = cfg["budgets"]
    est_in = exp["est_input_tokens"]
    est_out = exp["est_output_tokens"]
    util = args.thinking_utilization

    print("=" * 78)
    print(" ANCHORED MINDS -- COST ESTIMATE")
    print(f" {datetime.now(timezone.utc).isoformat()}")
    print("=" * 78)
    print(f" Design: {n_biases} biases x {n_items} items x {N_CONDITIONS} "
          f"conditions x {n_samples} samples/cell")
    print(f" Budgets: {budgets} (non-reasoning models: [0] only)")
    print(f" Token heuristics: ~{est_in} in / ~{est_out} out per call, "
          f"thinking utilization {util:.0%}")
    print("-" * 78)

    header = (f" {'model':<28}{'budgets':>8}{'calls':>9}"
              f"{'in tok (M)':>12}{'out tok (M)':>13}{'USD':>10}")
    print(header)
    print("-" * 78)

    rows = []
    total_calls, total_cost = 0, 0.0
    items_per_budget = n_biases * n_items * N_CONDITIONS * n_samples

    for m in cfg["models"]:
        model_budgets = budgets if m["reasoning"] else [0]
        calls = items_per_budget * len(model_budgets)

        in_tok = calls * est_in
        # visible answer tokens for every call + consumed thinking tokens
        out_tok = calls * est_out + sum(
            items_per_budget * b * util for b in model_budgets if b > 0)

        cost = (in_tok / 1e6) * m["price_in"] + (out_tok / 1e6) * m["price_out"]

        print(f" {m['name']:<28}{len(model_budgets):>8}{calls:>9,}"
              f"{in_tok / 1e6:>12.2f}{out_tok / 1e6:>13.2f}{cost:>10.2f}")

        rows.append({
            "model": m["name"], "model_id": m["model_id"],
            "reasoning": m["reasoning"], "n_budgets": len(model_budgets),
            "calls": calls, "input_tokens": in_tok, "output_tokens": int(out_tok),
            "price_in_per_M": m["price_in"], "price_out_per_M": m["price_out"],
            "est_cost_usd": round(cost, 2),
        })
        total_calls += calls
        total_cost += cost

    print("-" * 78)
    print(f" {'TOTAL':<28}{'':>8}{total_calls:>9,}{'':>12}{'':>13}"
          f"{total_cost:>10.2f}")
    print("=" * 78)

    # --- sanity warnings -----------------------------------------------------
    warnings = []
    if total_cost > 150:
        warnings.append(f"Estimated cost ${total_cost:,.0f} exceeds the ~$100 "
                        "solo budget -- consider fewer samples/cell, fewer "
                        "budgets, or cheaper models.")
    if total_calls > 30000:
        warnings.append(f"{total_calls:,} calls is a lot for the Day-2 window "
                        "-- check provider rate limits.")
    warnings.append("Prices in config.yaml are PLACEHOLDERS -- verify against "
                    "provider pricing pages before Day 0 sign-off.")
    print("\n WARNINGS / NOTES:")
    for w in warnings:
        print(f"  ! {w}")

    # --- save CSV -------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "cost_estimate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n CSV saved -> {csv_path}")

    # --- save TXT ---------------------------------------------------------------
    txt_path = out_dir / "cost_estimate.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"ANCHORED MINDS -- COST ESTIMATE "
                f"({datetime.now(timezone.utc).isoformat()})\n")
        f.write(f"Design grid: {n_biases} biases x {n_items} items x "
                f"{N_CONDITIONS} conditions x {n_samples} samples/cell\n")
        f.write(f"Budgets: {budgets}; thinking utilization {util:.0%}\n\n")
        f.write(f"{'model':<30}{'calls':>10}{'est_cost_usd':>15}\n")
        for r in rows:
            f.write(f"{r['model']:<30}{r['calls']:>10,}"
                    f"{r['est_cost_usd']:>15.2f}\n")
        f.write(f"\nTOTAL calls: {total_calls:,}\n")
        f.write(f"TOTAL est. cost: ${total_cost:,.2f}\n\n")
        for w in warnings:
            f.write(f"! {w}\n")
    print(f" TXT saved -> {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
