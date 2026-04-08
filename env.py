"""
env.py — Ward–population coupled AMR dynamics for stewardship OpenEnv.

Patient resistance and outcomes are tied to ward-level reservoirs (environmental /
nosocomial pressure). Reserve-antibiotic use adds collateral selection pressure on
the whole ward, not only on the treated isolate.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from models import (
    PrescriptionAction,
    ResetResult,
    StepResult,
    WardObservation,
    PatientState,
)

DRUG_KEYS = ("amoxicillin", "ciprofloxacin", "meropenem")


class PKPDModel:
    """Simple PK/PD + selection pressure; efficacy uses patient + ward resistance."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def evaluate_treatment(
        self,
        severity: float,
        drug: str,
        duration: int,
        patient_res: float,
        ward_res: float,
        task_id: str,
    ) -> tuple[float, float, bool]:
        """
        Returns new severity, incremental ward resistance pressure for this drug, is_dead.
        """
        if drug == "none":
            if task_id == "easy":
                lo, hi = 0.10, 0.42
            elif task_id == "medium":
                lo, hi = 0.16, 0.58
            else:
                lo, hi = 0.22, 0.78
            new_sev = min(10.0, severity + self.rng.uniform(lo, hi))
            return new_sev, 0.0, new_sev >= 10.0

        effective_res = min(1.0, 0.62 * patient_res + 0.38 * ward_res)
        efficacy = (1.0 - effective_res) * (duration ** 0.85) * self.rng.uniform(0.88, 1.12)

        eff_threshold = 0.44 if task_id == "easy" else (0.47 if task_id == "medium" else 0.50)
        if efficacy > severity * eff_threshold:
            new_sev = max(0.0, severity - efficacy * 0.95)
        else:
            bump_hi = 0.72 if task_id == "easy" else (0.82 if task_id == "medium" else 0.95)
            new_sev = min(10.0, severity + self.rng.uniform(0.12, bump_hi))

        if patient_res < 0.82:
            # Lower ward-selection pressure on medium/hard so the episode
            # is solvable without immediately driving the ward to 1.0 AMR.
            scale = (
                0.62
                if task_id == "easy"
                else (0.60 if task_id == "medium" else 0.78)
            )
            pressure = duration * scale * (0.010 + 0.022 * (1.0 - patient_res))
        else:
            pressure = duration * (
                0.032
                if task_id == "easy"
                else (0.028 if task_id == "medium" else 0.038)
            )

        return new_sev, pressure, new_sev >= 10.0


class AntibioticEnv:
    def __init__(self):
        self.rng = random.Random()
        self.task_id = "easy"
        self.ward_res: Dict[str, float] = {k: 0.0 for k in DRUG_KEYS}
        self.patients: List[Dict[str, Any]] = []
        self.step_count = 0
        self.max_steps = 30
        self.pkpd = PKPDModel(self.rng)
        self.cured = 0
        self.deaths = 0
        self.n_patients_initial = 0
        self.mero_patient_days = 0
        self.mero_budget = 999

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> ResetResult:
        if seed is not None:
            self.rng.seed(seed)
        self.task_id = task_id
        self.step_count = 0
        self.cured = 0
        self.deaths = 0
        self.mero_patient_days = 0

        if task_id == "easy":
            n_pat = 1
            self.max_steps = 12
            base_res = 0.12
            self.mero_budget = 999
        elif task_id == "medium":
            n_pat = 5
            self.max_steps = 32
            base_res = 0.34
            self.mero_budget = 999
        else:
            n_pat = 10
            self.max_steps = 44
            base_res = 0.58
            self.mero_budget = 14

        self.n_patients_initial = n_pat

        self.ward_res = {
            "amoxicillin": min(0.92, base_res + self.rng.uniform(-0.09, 0.09)),
            "ciprofloxacin": min(
                0.92, base_res * 0.72 + self.rng.uniform(-0.08, 0.1)
            ),
            "meropenem": min(0.35, 0.04 + self.rng.uniform(0.0, 0.06)),
        }

        self.patients = []
        for i in range(n_pat):
            noise = {k: self.rng.uniform(-0.08, 0.12) for k in DRUG_KEYS}
            self.patients.append(
                {
                    "id": f"P{i}",
                    "sev": self.rng.uniform(2.8, 7.2),
                    "cleared": False,
                    "dead": False,
                    "res": {
                        k: float(min(1.0, max(0.0, self.ward_res[k] + noise[k])))
                        for k in DRUG_KEYS
                    },
                }
            )

        return ResetResult(observation=self._obs(), info=self._grading_snapshot())

    def _living_indices(self) -> List[int]:
        return [i for i, p in enumerate(self.patients) if not p["cleared"] and not p["dead"]]

    def _mean_severity(self) -> float:
        idx = self._living_indices()
        if not idx:
            return 0.0
        return sum(self.patients[i]["sev"] for i in idx) / len(idx)

    def _population_res_index(self) -> float:
        return sum(self.ward_res[k] for k in DRUG_KEYS) / len(DRUG_KEYS)

    def _apply_ward_physics(self, delta_extra: Optional[Dict[str, float]] = None) -> None:
        if self.task_id == "easy":
            drift = 0.0028
        elif self.task_id == "medium":
            drift = 0.0040
        else:
            drift = 0.0095
        for k in DRUG_KEYS:
            self.ward_res[k] = min(1.0, self.ward_res[k] + drift)
        if delta_extra:
            for k, v in delta_extra.items():
                self.ward_res[k] = min(1.0, self.ward_res.get(k, 0.0) + v)

    def _sync_patient_resistance_from_ward(self) -> None:
        alpha = 0.09 if self.task_id == "easy" else (0.08 if self.task_id == "medium" else 0.12)
        for p in self.patients:
            if p["cleared"] or p["dead"]:
                continue
            for k in DRUG_KEYS:
                w, r = self.ward_res[k], p["res"][k]
                jitter = self.rng.uniform(-0.012, 0.012)
                p["res"][k] = float(
                    min(1.0, max(0.0, r + alpha * (w - r) + jitter))
                )

    def _grading_snapshot(self) -> Dict[str, Any]:
        living = self._living_indices()
        mean_sev = (
            sum(self.patients[i]["sev"] for i in living) / len(living) if living else 0.0
        )
        return {
            "task_id": self.task_id,
            "cured": self.cured,
            "deaths": self.deaths,
            "n_patients": self.n_patients_initial,
            "ward_res": {k: float(self.ward_res[k]) for k in DRUG_KEYS},
            "population_resistance_index": float(self._population_res_index()),
            "meropenem_patient_days": int(self.mero_patient_days),
            "meropenem_budget": int(self.mero_budget),
            "steps": self.step_count,
            "mean_severity_living": float(mean_sev),
        }

    def step(self, action: PrescriptionAction) -> StepResult:
        self.step_count += 1
        ward_sum_start = sum(self.ward_res[k] for k in DRUG_KEYS)
        mean_sev_start = self._mean_severity()
        cleared_start = sum(1 for p in self.patients if p["cleared"])
        deaths_start = sum(1 for p in self.patients if p["dead"])

        collateral: Dict[str, float] = {}

        target = next((p for p in self.patients if p["id"] == action.patient_id), None)
        if target and not target["cleared"] and not target["dead"]:
            drug = action.drug_choice
            if drug != "none":
                w_dr = self.ward_res[drug]
                p_dr = target["res"][drug]
                new_sev, pressure, dead = self.pkpd.evaluate_treatment(
                    target["sev"],
                    drug,
                    action.duration_days,
                    p_dr,
                    w_dr,
                    self.task_id,
                )
                target["sev"] = new_sev
                target["dead"] = dead
                self.ward_res[drug] = min(1.0, self.ward_res[drug] + pressure)
                if drug == "meropenem":
                    self.mero_patient_days += int(action.duration_days)
                    coll_scale = 0.60 if self.task_id == "easy" else (0.55 if self.task_id == "medium" else 1.0)
                    collateral = {
                        "amoxicillin": 0.0035 * coll_scale * action.duration_days,
                        "ciprofloxacin": 0.0035 * coll_scale * action.duration_days,
                    }
            else:
                new_sev, _, dead = self.pkpd.evaluate_treatment(
                    target["sev"], "none", 1, 0.0, 0.0, self.task_id
                )
                target["sev"] = new_sev
                target["dead"] = dead

        for p in self.patients:
            if p["id"] == action.patient_id or p["cleared"] or p["dead"]:
                continue
            new_sev, _, dead = self.pkpd.evaluate_treatment(
                p["sev"], "none", 1, 0.0, 0.0, self.task_id
            )
            p["sev"] = new_sev
            p["dead"] = dead

        for p in self.patients:
            if p["dead"]:
                p["sev"] = 10.0
                continue
            if p["sev"] <= 0.55:
                p["cleared"] = True
            elif p["sev"] >= 10.0:
                p["dead"] = True
                p["sev"] = 10.0

        self.cured = sum(1 for p in self.patients if p["cleared"])
        self.deaths = sum(1 for p in self.patients if p["dead"])

        new_cures = self.cured - cleared_start
        new_deaths = self.deaths - deaths_start

        self._apply_ward_physics(collateral if collateral else None)
        self._sync_patient_resistance_from_ward()

        reward = 0.0
        if new_cures > 0:
            reward += 0.32 * new_cures
        if new_deaths > 0:
            reward -= 0.42 * new_deaths

        if mean_sev_start > 1e-6:
            mean_sev_end = self._mean_severity()
            if mean_sev_end < mean_sev_start:
                reward += 0.55 * (mean_sev_start - mean_sev_end) / 10.0

        ward_sum_end = sum(self.ward_res[k] for k in DRUG_KEYS)
        dw = max(0.0, ward_sum_end - ward_sum_start)
        reward -= 0.18 * (dw / float(len(DRUG_KEYS)))

        pop_idx = self._population_res_index()
        reward -= 0.12 * pop_idx

        if action.drug_choice == "meropenem":
            reward -= 0.045 * (action.duration_days ** 0.9)

        if self.task_id == "hard" and self.mero_patient_days > self.mero_budget:
            reward -= 0.06 * (self.mero_patient_days - self.mero_budget)

        reward = max(-1.0, min(1.0, float(reward)))

        done = self.step_count >= self.max_steps or all(
            p["cleared"] or p["dead"] for p in self.patients
        )

        return StepResult(
            observation=self._obs(),
            reward=reward,
            done=done,
            info=self._grading_snapshot(),
        )

    def _obs(self) -> WardObservation:
        patients = [
            PatientState(
                patient_id=p["id"],
                severity_score=p["sev"] if not p["dead"] else 10.0,
                infection_cleared=p["cleared"],
                res_amox=p["res"]["amoxicillin"],
                res_cipro=p["res"]["ciprofloxacin"],
                res_mero=p["res"]["meropenem"],
            )
            for p in self.patients
        ]

        return WardObservation(
            step_count=self.step_count,
            max_steps=self.max_steps,
            patients=patients,
            ward_amox_resistance=self.ward_res["amoxicillin"],
            ward_cipro_resistance=self.ward_res["ciprofloxacin"],
            ward_mero_resistance=self.ward_res["meropenem"],
            population_resistance_index=self._population_res_index(),
            meropenem_cumulative_patient_days=int(self.mero_patient_days),
            meropenem_patient_day_budget=int(self.mero_budget),
            total_patients_cured=self.cured,
            total_patient_deaths=self.deaths,
            budget_remaining=max(
                0.0, 1.0 - (self.step_count / max(1, self.max_steps))
            ),
        )

    def state(self) -> Dict[str, Any]:
        return self._grading_snapshot()
