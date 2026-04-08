"""
Inference Script — Antibiotic Stewardship OpenEnv
===================================================
MANDATORY env vars (set by Scaler grader):
    API_BASE_URL      LLM endpoint
    MODEL_NAME        Model identifier
    HF_TOKEN          Hugging Face / API key  (also checks API_KEY)
    LOCAL_IMAGE_NAME  Docker image name (not used — we use HTTP ENV_URL)
    ENV_URL           URL of the running environment server

STDOUT FORMAT (parsed by Scaler grader):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional

import requests
from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL   = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME     = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
API_KEY        = os.getenv("HF_TOKEN")     or os.getenv("API_KEY") or ""
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL        = os.getenv("ENV_URL", "http://127.0.0.1:7860")
BASELINE_SEED  = int(os.getenv("BASELINE_SEED", "42"))
BENCHMARK      = "antibiotic-stewardship"

TASK_ORDER = ("easy", "medium", "hard")

# ── Mandatory stdout helpers ───────────────────────────────────────────────────
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Heuristic agent (runs without LLM) ────────────────────────────────────────
def heuristic_action(obs: dict[str, Any]) -> dict[str, Any]:
    patients = [
        p for p in (obs.get("patients") or [])
        if not p.get("infection_cleared") and float(p.get("severity_score", 0)) < 10.0
    ]
    if not patients:
        return {"patient_id": "P0", "drug_choice": "none", "duration_days": 1}

    patients.sort(key=lambda x: float(x.get("severity_score", 0)), reverse=True)
    p = patients[0]
    pid = p["patient_id"]

    mero_budget = int(obs.get("meropenem_patient_day_budget") or 999)
    mero_used   = int(obs.get("meropenem_cumulative_patient_days") or 0)
    remaining   = max(0, mero_budget - mero_used)
    ward_amox   = float(obs.get("ward_amox_resistance") or 0.0)
    ward_cip    = float(obs.get("ward_cipro_resistance") or 0.0)
    ward_mero   = float(obs.get("ward_mero_resistance") or 0.0)
    ra, rc, rm  = float(p["res_amox"]), float(p["res_cipro"]), float(p["res_mero"])
    sev         = float(p.get("severity_score", 0))
    pop         = float(obs.get("population_resistance_index", 0.0))
    hard_mode   = mero_budget <= 100

    am_score = ra + 0.45 * ward_amox
    ci_score = rc + 0.45 * ward_cip
    best_narrow       = "amoxicillin" if am_score <= ci_score else "ciprofloxacin"
    best_narrow_score = min(am_score, ci_score)

    if hard_mode:
        use_mero = (remaining >= 2 and sev >= 7.5
                    and best_narrow_score > 0.78 and rm < 0.78 and ward_mero < 0.85)
        drug     = "meropenem" if use_mero else best_narrow
        duration = min(4, remaining, 3 + (1 if sev >= 8.5 else 0)) if drug == "meropenem" else (5 if sev < 7.0 else 6)
    else:
        use_mero = (sev >= 8.8 and best_narrow_score > 0.88 and rm < 0.75 and ward_mero < 0.85)
        drug     = "meropenem" if use_mero else best_narrow
        duration = 3 if drug == "meropenem" else (4 if sev < 7.0 else 5)
        if pop > 0.65:
            duration = min(duration, 4 if drug != "meropenem" else 2)

    return {"patient_id": pid, "drug_choice": drug, "duration_days": max(1, min(14, int(duration)))}


# ── LLM agent ─────────────────────────────────────────────────────────────────
def obs_to_prompt(obs: dict[str, Any]) -> str:
    patients = obs.get("patients") or []
    lines = [
        f"step={obs.get('step_count')}/{obs.get('max_steps')}",
        f"ward_res amox={obs.get('ward_amox_resistance'):.3f} cipro={obs.get('ward_cipro_resistance'):.3f} mero={obs.get('ward_mero_resistance'):.3f}",
        f"population_resistance_index={obs.get('population_resistance_index', 0):.3f}",
        f"meropenem_cumulative_patient_days={obs.get('meropenem_cumulative_patient_days', 0)} budget={obs.get('meropenem_patient_day_budget', 0)}",
        "patients:",
    ]
    for p in patients:
        if p.get("infection_cleared") or p.get("severity_score", 0) >= 10.0:
            continue
        lines.append(
            f"  - id={p['patient_id']} sev={p['severity_score']:.2f} "
            f"res_amox={p['res_amox']:.2f} res_cipro={p['res_cipro']:.2f} res_mero={p['res_mero']:.2f}"
        )
    lines.append('Respond with JSON only: {"patient_id":"<id>","drug_choice":"none|amoxicillin|ciprofloxacin|meropenem","duration_days":<1-14 int>}')
    return "\n".join(lines)


def llm_action(client: OpenAI, model: str, obs: dict[str, Any]) -> dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a hospital antimicrobial stewardship assistant. "
                    "Minimize reserve-antibiotic pressure while clearing infections. "
                    "Output JSON only."
                )},
                {"role": "user", "content": obs_to_prompt(obs)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw  = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        return {
            "patient_id":   str(data.get("patient_id", "P0")),
            "drug_choice":  str(data.get("drug_choice", "none")),
            "duration_days": int(data.get("duration_days", 5)),
        }
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return heuristic_action(obs)


# ── Graders ────────────────────────────────────────────────────────────────────
from graders.easy_grader   import SinglePatientGrader
from graders.medium_grader import WardManagementGrader
from graders.hard_grader   import ReserveBudgetGrader

GRADERS = {
    "easy":   SinglePatientGrader(),
    "medium": WardManagementGrader(),
    "hard":   ReserveBudgetGrader(),
}


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(
    task_id: str,
    seed: int,
    client: Optional[OpenAI],
    model: str,
) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, model=model)

    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=60)
        r.raise_for_status()
        obs       = r.json()["observation"]
        last_info: dict[str, Any] = {}

        for step_num in range(1, 200):
            # Check if any treatable patients remain
            sick = [
                p for p in (obs.get("patients") or [])
                if not p.get("infection_cleared") and float(p.get("severity_score", 0)) < 10.0
            ]
            if not sick:
                break

            action = llm_action(client, model, obs) if client else heuristic_action(obs)

            error_msg: Optional[str] = None
            try:
                resp = requests.post(f"{ENV_URL}/step", json=action, timeout=60)
                resp.raise_for_status()
                body    = resp.json()
                obs     = body.get("observation", obs)
                reward  = float(body.get("reward", 0.0))
                done    = bool(body.get("done", False))
                last_info = body.get("info") or {}
            except Exception as exc:
                reward    = 0.0
                done      = True
                error_msg = str(exc)

            rewards.append(reward)
            steps_taken = step_num

            action_str = (
                f"{action.get('patient_id','?')}:"
                f"{action.get('drug_choice','none')}:"
                f"{action.get('duration_days',1)}d"
            )
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Get final grader score
        try:
            st = requests.get(f"{ENV_URL}/state", timeout=30).json()
            if isinstance(st, dict) and "cured" in st:
                last_info = st
        except Exception:
            pass

        score   = float(GRADERS[task_id].score(last_info))
        score   = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[*TASK_ORDER, "all"],
        default="all",
        help="Which task to run (default: all).",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--seed",  type=int, default=BASELINE_SEED)
    args = parser.parse_args()

    tasks = list(TASK_ORDER) if args.task == "all" else [args.task]
    model = args.model or MODEL_NAME

    # Scaler grader always injects API_BASE_URL and API_KEY — use os.environ[] as required
    client: Optional[OpenAI] = None
    try:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except KeyError:
        # Not running under Scaler grader — fall back to local env vars for testing
        _key  = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
        _base = os.getenv("API_BASE_URL", API_BASE_URL)
        if _key:
            try:
                client = OpenAI(base_url=_base, api_key=_key)
            except Exception as exc:
                print(f"[WARN] Could not create OpenAI client: {exc}. Using heuristic mode.", flush=True)
        else:
            print("[WARN] No API key found. Using heuristic mode.", flush=True)

    for task_id in tasks:
        run_episode(task_id=task_id, seed=args.seed, client=client, model=model)


if __name__ == "__main__":
    main()
