#!/usr/bin/env python3
"""
01_data_collection/run_experiments.py

The main experimental harness. Runs the full design grid

    model x reasoning-budget x battery-item x condition x sample

against the OpenRouter API (async, retry/backoff, CHECKPOINTED: kill it and
rerun -- completed calls are skipped automatically).

Conditions:
  control               the unbiased prompt
  treatment             the bias-manipulated prompt
  treatment_verbalized  (RQ4, Anchoring items only) treatment prompt with an
                        instruction to explicitly restate any numbers before
                        answering

Budget semantics:
  budget 0  -> non-reasoning models: plain call.
               reasoning models: request thinking disabled; if the provider
               rejects that with HTTP 400, the call is retried once without
               the reasoning parameter. For models flagged
               cannot_disable_thinking these cells are excluded downstream.
  budget >0 -> reasoning models only: thinking ceiling of that many tokens,
               with max_tokens = budget + 2 x max_output_tokens so that long
               deliberations do not truncate the visible answer.

Usage:
  python 01_data_collection/run_experiments.py --smoke-test   # small live run
  python 01_data_collection/run_experiments.py                # full grid
  python 01_data_collection/run_experiments.py --model deepseek-r1
  python 01_data_collection/run_experiments.py --dry-run      # no API calls;
                                                              # simulated
                                                              # responses

Saves (in 01_data_collection/outputs/raw_responses/):
  {model}__b{budget}.jsonl    one line per completed call (checkpoint files)
  run_manifest.json           grid definition + live counters
  run_log.txt                 condensed run log
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANSWER_INSTRUCTION = (
    "\n\nRespond with the single option you choose, in exactly this format:\n"
    "Answer: Option <number>"
)
VERBALIZE_INSTRUCTION = (
    "\n\nBefore answering, explicitly restate every specific number mentioned "
    "in the prompt above. Then respond with the single option you choose, in "
    "exactly this format:\nAnswer: Option <number>"
)
QUICK_PARSE_RE = re.compile(r"Answer:\s*Option\s*(\d+)", re.IGNORECASE)
FALLBACK_PARSE_RE = re.compile(r"Option\s*(\d+)", re.IGNORECASE)

LOG_LINES = []


def log(line: str = "") -> None:
    print(line, flush=True)
    LOG_LINES.append(line)


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------
def build_tasks(cfg, battery, models, budgets_override=None):
    """Enumerate every (model, budget, item, condition, sample) cell."""
    n_samples = cfg["experiment"]["n_samples_per_cell"]
    tasks = []
    for m in models:
        budgets = (budgets_override if budgets_override is not None
                   else (cfg["budgets"] if m["reasoning"] else [0]))
        for b in budgets:
            for item in battery:
                conditions = ["control", "treatment"]
                if item["bias"] == "Anchoring":
                    conditions.append("treatment_verbalized")  # RQ4
                for cond in conditions:
                    for s in range(n_samples):
                        tasks.append({
                            "task_id": f"{m['name']}|b{b}|{item['item_id']}|{cond}|s{s}",
                            "model": m, "budget": b, "item": item,
                            "condition": cond, "sample_idx": s,
                        })
    return tasks


def task_prompt(task):
    item, cond = task["item"], task["condition"]
    if cond == "control":
        return item["control_text"] + ANSWER_INSTRUCTION
    if cond == "treatment":
        return item["treatment_text"] + ANSWER_INSTRUCTION
    if cond == "treatment_verbalized":
        return item["treatment_text"] + VERBALIZE_INSTRUCTION
    raise ValueError(cond)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def build_payload(cfg, task):
    m, b = task["model"], task["budget"]
    max_out = cfg["experiment"]["max_output_tokens"]
    payload = {
        "model": m["model_id"],
        "messages": [{"role": "user", "content": task_prompt(task)}],
        "temperature": cfg["experiment"]["temperature"],
        # answer allowance is doubled for thinking calls: prevents the
        # thinking phase from starving the visible answer (truncation)
        "max_tokens": (b + 2 * max_out) if b > 0 else max_out,
    }
    if m["reasoning"]:
        if b > 0:
            payload["reasoning"] = {"max_tokens": b}
        else:
            payload["reasoning"] = {"enabled": False}
    return payload


async def call_api(client, cfg, task, sem, dry_run=False):
    """One API call with retry/backoff. Returns a result record dict."""
    exp = cfg["experiment"]
    rec = {
        "task_id": task["task_id"],
        "model": task["model"]["name"],
        "model_id": task["model"]["model_id"],
        "reasoning": task["model"]["reasoning"],
        "pair": task["model"].get("pair"),
        "budget": task["budget"],
        "item_id": task["item"]["item_id"],
        "bias": task["item"]["bias"],
        "condition": task["condition"],
        "sample_idx": task["sample_idx"],
        "n_options": task["item"]["n_options"],
        "k": task["item"]["k"],
        "flip_treatment": task["item"]["flip_treatment"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if dry_run:  # simulate a plausible response for pipeline testing
        await asyncio.sleep(0.001)
        n = task["item"]["n_options"]
        rec.update(status="ok", latency_s=0.0,
                   response_text=f"Answer: Option {random.randint(1, n)}",
                   quick_parse=None, usage={"prompt_tokens": 400,
                                            "completion_tokens": 20})
        rec["quick_parse"] = int(QUICK_PARSE_RE.search(
            rec["response_text"]).group(1))
        return rec

    payload = build_payload(cfg, task)
    last_err = None
    async with sem:
        for attempt in range(exp["max_retries"] + 1):
            t0 = time.monotonic()
            try:
                r = await client.post("/chat/completions", json=payload,
                                      timeout=exp["request_timeout_s"])
                latency = time.monotonic() - t0
                if r.status_code == 200:
                    data = r.json()
                    # OpenRouter can return 200 with an embedded error object
                    if "error" in data and not data.get("choices"):
                        last_err = f"api_error_in_200: {str(data['error'])[:200]}"
                    else:
                        text = (data["choices"][0]["message"].get("content")
                                or "")
                        qp = QUICK_PARSE_RE.search(text) \
                            or FALLBACK_PARSE_RE.search(text)
                        rec.update(
                            status="ok", latency_s=round(latency, 2),
                            response_text=text,
                            quick_parse=int(qp.group(1)) if qp else None,
                            usage=data.get("usage", {}),
                            provider=data.get("provider"),
                        )
                        return rec
                elif r.status_code == 400 and task["budget"] == 0 \
                        and task["model"]["reasoning"] \
                        and "reasoning" in payload:
                    # provider can't disable thinking -> retry once without
                    payload.pop("reasoning")
                    last_err = "reasoning_disable_rejected"
                    continue
                elif r.status_code in (429, 500, 502, 503, 529):
                    last_err = f"http_{r.status_code}"
                else:
                    rec.update(status="failed", latency_s=round(latency, 2),
                               error=f"http_{r.status_code}: {r.text[:200]}")
                    return rec
            except Exception as e:  # timeouts, connection errors
                last_err = repr(e)[:200]
            # backoff before retry
            await asyncio.sleep(min(2 ** attempt + random.random(), 30))

    rec.update(status="failed", error=f"retries_exhausted: {last_err}")
    return rec


# ---------------------------------------------------------------------------
# Runner with checkpointing
# ---------------------------------------------------------------------------
def load_completed(raw_dir):
    """Read all existing JSONL checkpoints -> set of completed task_ids."""
    done = set()
    for f in raw_dir.glob("*__b*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    done.add(rec["task_id"])
            except json.JSONDecodeError:
                continue  # ignore a possibly truncated last line
    return done


async def run(cfg, tasks, raw_dir, dry_run=False):
    import httpx
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    exp = cfg["experiment"]
    sem = asyncio.Semaphore(exp["concurrency"])
    key = os.getenv(cfg["providers"]["openrouter"]["env_key"], "")
    headers = {"Authorization": f"Bearer {key}",
               "HTTP-Referer": "https://localhost",
               "X-Title": "anchored-minds"}
    base = cfg["providers"]["openrouter"]["base_url"]

    files = {}   # (model, budget) -> open file handle

    def fh(model_name, budget):
        key_ = (model_name, budget)
        if key_ not in files:
            files[key_] = open(raw_dir / f"{model_name}__b{budget}.jsonl",
                               "a", encoding="utf-8")
        return files[key_]

    stats = {"ok": 0, "failed": 0, "parse_ok": 0,
             "in_tok": 0, "out_tok": 0}
    t_start = time.monotonic()
    n_total = len(tasks)

    async with httpx.AsyncClient(base_url=base, headers=headers) as client:
        pending = [call_api(client, cfg, t, sem, dry_run) for t in tasks]
        for i, coro in enumerate(asyncio.as_completed(pending), start=1):
            rec = await coro
            f = fh(rec["model"], rec["budget"])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if rec["status"] == "ok":
                stats["ok"] += 1
                if rec.get("quick_parse") is not None:
                    stats["parse_ok"] += 1
                u = rec.get("usage") or {}
                stats["in_tok"] += u.get("prompt_tokens", 0) or 0
                stats["out_tok"] += u.get("completion_tokens", 0) or 0
            else:
                stats["failed"] += 1
            if i % 25 == 0 or i == n_total:
                rate = stats["parse_ok"] / max(stats["ok"], 1) * 100
                elapsed = time.monotonic() - t_start
                eta = elapsed / i * (n_total - i)
                print(f"  [{i:>6}/{n_total}] ok={stats['ok']} "
                      f"failed={stats['failed']} parse-ok={rate:.1f}% "
                      f"| {elapsed:6.0f}s elapsed, ~{eta:5.0f}s left",
                      flush=True)

    for f in files.values():
        f.close()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the experimental grid")
    parser.add_argument("--smoke-test", action="store_true",
                        help="tiny live run: 2 biases x 2 items x 2 samples, "
                             "2 models, budgets [0, max]")
    parser.add_argument("--dry-run", action="store_true",
                        help="no API calls; writes simulated responses "
                             "(pipeline test)")
    parser.add_argument("--model", type=str, default=None,
                        help="run only this model (config 'name')")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw_dir = ROOT / cfg["paths"]["raw_responses_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    battery_path = ROOT / cfg["paths"]["battery_dir"] / "final_battery.json"
    if not battery_path.exists():
        log("[ERROR] No frozen battery found. Run build_test_battery.py first.")
        return 1
    battery = json.loads(battery_path.read_text())
    models = cfg["models"]

    log("=" * 70)
    log(" ANCHORED MINDS -- RUN EXPERIMENTS")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    mode = ("DRY-RUN" if args.dry_run else
            "SMOKE-TEST" if args.smoke_test else "FULL GRID")
    log(f" Mode      : {mode}")
    log("=" * 70)

    budgets_override = None
    if args.smoke_test:
        smoke_biases = cfg["biases"][:2]
        battery = [it for it in battery if it["bias"] in smoke_biases][:4]
        # keep only first 2 items per bias
        seen = {}
        battery = [it for it in battery
                   if seen.setdefault(it["bias"], []).append(it["item_id"])
                   or len(seen[it["bias"]]) <= 2]
        models = models[:2]
        cfg["experiment"]["n_samples_per_cell"] = 2
        budgets_override = [0, max(cfg["budgets"])]
        log(f" Smoke scope: biases={smoke_biases}, "
            f"models={[m['name'] for m in models]}, budgets={budgets_override}")
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            log(f"[ERROR] Unknown model '{args.model}'. Valid: "
                f"{[m['name'] for m in cfg['models']]}")
            return 1

    tasks = build_tasks(cfg, battery, models, budgets_override)
    done = load_completed(raw_dir)
    todo = [t for t in tasks if t["task_id"] not in done]
    log(f"\n Grid size : {len(tasks):,} calls "
        f"({len(done):,} already completed, {len(todo):,} to run)")
    if not todo:
        log(" Nothing to do -- grid already complete \u2713")
        return 0

    if not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if not os.getenv(cfg["providers"]["openrouter"]["env_key"]):
            log("[ERROR] OPENROUTER_API_KEY not set. Fill .env first.")
            return 1

    stats = asyncio.run(run(cfg, todo, raw_dir, dry_run=args.dry_run))

    # ------------------------------------------------------------- manifest --
    parse_rate = stats["parse_ok"] / max(stats["ok"], 1) * 100
    manifest = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "grid_total": len(tasks),
        "completed_before": len(done),
        "ran_now": {"ok": stats["ok"], "failed": stats["failed"]},
        "quick_parse_rate_pct": round(parse_rate, 2),
        "tokens": {"input": stats["in_tok"], "output": stats["out_tok"]},
        "models": [m["name"] for m in models],
        "n_battery_items": len(battery),
    }
    (raw_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    log("\n" + "=" * 70)
    log(f" RESULT: {stats['ok']} ok, {stats['failed']} failed "
        f"| quick parse rate {parse_rate:.1f}%")
    log(f" Tokens: {stats['in_tok']:,} in / {stats['out_tok']:,} out")
    if parse_rate < 95:
        log(" [!] Quick parse rate below 95% -- inspect responses before "
            "scaling up.")
    log(f" Manifest -> {raw_dir.relative_to(ROOT)}/run_manifest.json")
    log(" Next step: python 02_data_processing/parse_responses.py")
    log("=" * 70)

    log_path = raw_dir / "run_log.txt"
    log_path.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
