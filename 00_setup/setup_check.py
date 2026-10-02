#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 00_setup/setup_check.py
=============================================================================
 Pre-flight verification. Run this FIRST, before spending any API money.

 Checks, in order:
   1. Python version and required packages are importable
   2. config.yaml loads and contains all required sections
   3. .env exists and every API key required by the model panel is set
   4. All output directories exist (creates any that are missing)
   5. (optional, --ping) one minimal live API call per provider to confirm
      the key actually authenticates

 Usage:
   python 00_setup/setup_check.py           # offline checks only
   python 00_setup/setup_check.py --ping    # + live 1-token API pings

 Saves:
   00_setup/setup_report.txt   (full check log with timestamps)
=============================================================================
"""

import argparse
import importlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- resolve repo root no matter where the script is called from -------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORT_LINES = []


def log(line: str = "") -> None:
    """Print to console AND buffer for the saved report."""
    print(line, flush=True)
    REPORT_LINES.append(line)


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "\u2713" if ok else "\u2717"
    log(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Anchored Minds pre-flight check")
    parser.add_argument("--ping", action="store_true",
                        help="also make one minimal live API call per provider")
    args = parser.parse_args()

    log("=" * 70)
    log(" ANCHORED MINDS -- SETUP CHECK")
    log(f" Repo root : {ROOT}")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log("=" * 70)

    all_ok = True

    # ------------------------------------------------------------------ 1 ---
    log("\n[1/5] Python & packages")
    py_ok = sys.version_info >= (3, 10)
    all_ok &= check(f"Python >= 3.10 (found {sys.version.split()[0]})", py_ok)

    required = ["yaml", "dotenv", "httpx", "pandas", "numpy", "scipy",
                "statsmodels", "matplotlib", "tqdm"]
    for pkg in required:
        try:
            importlib.import_module(pkg)
            check(f"import {pkg}", True)
        except ImportError as e:
            all_ok &= check(f"import {pkg}", False, str(e))

    # ------------------------------------------------------------------ 2 ---
    log("\n[2/5] config.yaml")
    cfg = None
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        all_ok &= check("config.yaml exists", False, f"not found at {cfg_path}")
    else:
        check("config.yaml exists", True)
        import yaml
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            check("config.yaml parses", True)
        except yaml.YAMLError as e:
            all_ok &= check("config.yaml parses", False, str(e))

    if cfg:
        for section in ["project", "paths", "experiment", "biases",
                        "budgets", "models", "providers"]:
            all_ok &= check(f"section '{section}' present", section in cfg)

        n_models = len(cfg.get("models", []))
        n_reason = sum(1 for m in cfg.get("models", []) if m.get("reasoning"))
        log(f"      -> {n_models} models configured "
            f"({n_reason} reasoning, {n_models - n_reason} non-reasoning)")
        log(f"      -> {len(cfg.get('biases', []))} biases | "
            f"budgets: {cfg.get('budgets')}")

        # every model must reference a configured provider
        provs = set(cfg.get("providers", {}).keys())
        for m in cfg.get("models", []):
            ok = m.get("provider") in provs
            all_ok &= check(f"model '{m.get('name')}' -> provider "
                            f"'{m.get('provider')}' configured", ok)

    # ------------------------------------------------------------------ 3 ---
    log("\n[3/5] Environment / API keys")
    from dotenv import load_dotenv
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        check(".env file found and loaded", True)
    else:
        check(".env file found", False,
              "copy .env.example -> .env and fill in your keys")
        all_ok = False

    needed_providers = set()
    if cfg:
        needed_providers = {m["provider"] for m in cfg.get("models", [])}
        for prov in sorted(needed_providers):
            key_name = cfg["providers"][prov]["env_key"]
            val = os.getenv(key_name, "")
            ok = bool(val) and not val.startswith("sk-...")
            masked = (val[:8] + "..." + val[-4:]) if ok and len(val) > 14 else "MISSING"
            all_ok &= check(f"{key_name} set ({prov})", ok, masked)

    # ------------------------------------------------------------------ 4 ---
    log("\n[4/5] Directory structure")
    expected_dirs = [
        "01_data_collection/outputs/battery",
        "01_data_collection/outputs/raw_responses",
        "02_data_processing/outputs",
        "03_codebook/outputs",
        "04_analysis/rq1_reasoning_vs_bias/outputs/figures",
        "04_analysis/rq2_dose_response/outputs/figures",
        "04_analysis/rq3_bias_taxonomy/outputs/figures",
        "04_analysis/rq4_verbalization/outputs/figures",
        "05_paper_assets/outputs",
    ]
    for d in expected_dirs:
        path = ROOT / d
        if path.exists():
            check(f"{d}/", True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            check(f"{d}/", True, "created")

    # ------------------------------------------------------------------ 5 ---
    log("\n[5/5] Live API pings" + ("" if args.ping else "  (skipped -- use --ping)"))
    if args.ping and cfg:
        import httpx
        for prov in sorted(needed_providers):
            pconf = cfg["providers"][prov]
            key = os.getenv(pconf["env_key"], "")
            if not key:
                all_ok &= check(f"ping {prov}", False, "no key, skipping")
                continue
            try:
                if prov == "anthropic":
                    r = httpx.post(
                        f"{pconf['base_url']}/messages",
                        headers={"x-api-key": key,
                                 "anthropic-version": "2023-06-01"},
                        json={"model": "claude-haiku-4-5",
                              "max_tokens": 1,
                              "messages": [{"role": "user", "content": "hi"}]},
                        timeout=30)
                else:  # openai-compatible (openrouter, openai)
                    r = httpx.get(f"{pconf['base_url']}/models",
                                  headers={"Authorization": f"Bearer {key}"},
                                  timeout=30)
                ok = r.status_code == 200
                all_ok &= check(f"ping {prov}", ok, f"HTTP {r.status_code}")
            except httpx.HTTPError as e:
                all_ok &= check(f"ping {prov}", False, repr(e))

    # ---------------------------------------------------------------- done --
    log("\n" + "=" * 70)
    if all_ok:
        log(" RESULT: ALL CHECKS PASSED \u2713")
        log(" Next step: python 00_setup/cost_estimator.py")
    else:
        log(" RESULT: SOME CHECKS FAILED \u2717 -- fix the items above and re-run.")
    log("=" * 70)

    report_path = Path(__file__).resolve().parent / "setup_report.txt"
    report_path.write_text("\n".join(REPORT_LINES) + "\n", encoding="utf-8")
    print(f"\nFull report saved -> {report_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
