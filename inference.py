"""
Baseline inference for antibiotic stewardship OpenEnv.
Uses OpenAI-compatible client (HF router, OpenAI, etc.) via env vars:
  API_BASE_URL, MODEL_NAME, HF_TOKEN (and/or OPENAI_API_KEY), ENV_URL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests
from openai import OpenAI

from graders.hard_grader import ReserveBudgetGrader
from graders.easy_grader import SinglePatientGrader
from graders.medium_grader import WardManagementGrader

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-server>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")
BASELINE_SEED = int(os.getenv("BASELINE_SEED", "42"))

TASK_ORDER = ("easy", "medium", "hard")
GRADERS = {
    "easy": SinglePatientGrader(),
    "medium": WardManagementGrader(),
    "hard": ReserveBudgetGrader(),
}


def log(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, sort_keys=True)}")
    sys.stdout.flush()


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
    lines.append(
        'Respond with JSON only: {"patient_id":"<id>","drug_choice":"none|amoxicillin|ciprofloxacin|meropenem","duration_days":<1-14 int>}'
    )
    return "\n".join(lines)


def heuristic_action(obs: dict[str, Any]) -> dict[str, Any]:
    patients = [
        p
        for p in (obs.get("patients") or [])
        if not p.get("infection_cleared") and p.get("severity_score", 0) < 10.0
    ]
    if not patients:
        return {"patient_id": "P0", "drug_choice": "none", "duration_days": 1}
    patients.sort(key=lambda x: float(x.get("severity_score", 0)), reverse=True)
    p = patients[0]
    pid = p["patient_id"]
    mero_budget = int(obs.get("meropenem_patient_day_budget") or 999)
    mero_used = int(obs.get("meropenem_cumulative_patient_days") or 0)
    ward_amox = float(obs.get("ward_amox_resistance") or 0.0)
    ward_cip = float(obs.get("ward_cipro_resistance") or 0.0)
    ward_mero = float(obs.get("ward_mero_resistance") or 0.0)

    ra, rc, rm = float(p["res_amox"]), float(p["res_cipro"]), float(p["res_mero"])
    sev = float(p.get("severity_score", 0))

    # Budget sentinel: easy/medium have large budget, hard has small budget.
    hard_mode = mero_budget <= 100
    remaining = max(0, mero_budget - mero_used)

    # Prefer the drug with best "effective resistance" estimate.
    # This is intentionally conservative: for medium we mostly avoid meropenem.
    am_score = ra + 0.45 * ward_amox
    ci_score = rc + 0.45 * ward_cip
    best_narrow = "amoxicillin" if am_score <= ci_score else "ciprofloxacin"
    best_narrow_score = min(am_score, ci_score)

    pop = float(obs.get("population_resistance_index", 0.0))

    if hard_mode:
        # Only use meropenem if both narrow options look bad and we still have reserve budget.
        use_meropenem = (
            remaining >= 2
            and sev >= 7.5
            and best_narrow_score > 0.78
            and rm < 0.78
            and ward_mero < 0.85
        )
        drug = "meropenem" if use_meropenem else best_narrow
        if drug == "meropenem":
            duration = min(4, remaining, 3 + (1 if sev >= 8.5 else 0))
        else:
            duration = 5 if sev < 7.0 else 6
    else:
        # Medium: avoid meropenem unless severity is extremely high *and*
        # narrow options are clearly failing.
        use_meropenem = (
            sev >= 8.8
            and best_narrow_score > 0.88
            and rm < 0.75
            and ward_mero < 0.85
        )
        drug = "meropenem" if use_meropenem else best_narrow
        duration = 4 if sev < 7.0 else 5
        if drug == "meropenem":
            duration = 3
        if pop > 0.65:
            duration = min(duration, 4 if drug != "meropenem" else 2)

    return {
        "patient_id": pid,
        "drug_choice": drug,
        "duration_days": max(1, min(14, int(duration))),
    }


def llm_action(client: OpenAI, model: str, obs: dict[str, Any]) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a hospital antimicrobial stewardship assistant. "
                    "Minimize reserve-antibiotic pressure while clearing infections. "
                    "Output JSON only."
                ),
            },
            {"role": "user", "content": obs_to_prompt(obs)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = completion.choices[0].message.content or "{}"
    data = json.loads(raw)
    return {
        "patient_id": str(data.get("patient_id", "P0")),
        "drug_choice": str(data.get("drug_choice", "none")),
        "duration_days": int(data.get("duration_days", 5)),
    }


def run_episode(
    task_id: str,
    seed: int,
    client: OpenAI | None,
    model: str,
) -> dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=60)
    r.raise_for_status()
    reset_body = r.json()
    obs = reset_body["observation"]
    total_reward = 0.0
    step_count = 0
    last_info: dict[str, Any] = {}

    while True:
        sick = [
            p
            for p in (obs.get("patients") or [])
            if not p.get("infection_cleared") and float(p.get("severity_score", 0)) < 10.0
        ]
        if not sick:
            break

        if client:
            try:
                action = llm_action(client, model, obs)
            except Exception:
                action = heuristic_action(obs)
        else:
            action = heuristic_action(obs)

        step_count += 1
        step = requests.post(f"{ENV_URL}/step", json=action, timeout=60)
        step.raise_for_status()
        body = step.json()
        obs = body.get("observation", obs)
        reward = float(body.get("reward", 0.0))
        done = bool(body.get("done", False))
        last_info = body.get("info") or {}
        total_reward += reward
        log(
            "STEP",
            {
                "action": action,
                "done": done,
                "reward": reward,
                "step": step_count,
                "task_id": task_id,
            },
        )
        if done:
            break

    try:
        st = requests.get(f"{ENV_URL}/state", timeout=30).json()
        if isinstance(st, dict) and "cured" in st:
            last_info = st
    except Exception:
        pass

    grade = float(GRADERS[task_id].score(last_info))
    return {
        "grader_score": grade,
        "info": last_info,
        "steps": step_count,
        "task_id": task_id,
        "total_reward": total_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[*TASK_ORDER, "all"],
        default="all",
        help="Which task to run (default: all).",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=BASELINE_SEED)
    args = parser.parse_args()

    tasks = list(TASK_ORDER) if args.task == "all" else [args.task]

    client: OpenAI | None = None
    if HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception:
            client = None

    log(
        "START",
        {
            "env_url": ENV_URL,
            "model": args.model,
            "seed": args.seed,
            "tasks": tasks,
            "use_llm": bool(client),
        },
    )

    results = []
    for tid in tasks:
        results.append(run_episode(tid, args.seed, client, args.model))

    by_task = {r["task_id"]: r["grader_score"] for r in results}
    mean_grade = sum(by_task.values()) / max(1, len(by_task))

    log(
        "END",
        {
            "grader_scores": by_task,
            "mean_grader_score": mean_grade,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
