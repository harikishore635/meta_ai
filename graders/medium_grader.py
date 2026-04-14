"""
Medium Grader — WardManagementGrader
=====================================
Evaluates the agent's ability to manage a multi-patient ward
while balancing clinical outcomes against population-level AMR.

Scoring breakdown:
  - 40% clinical outcomes     (cure rate minus death penalty)
  - 30% AMR containment       (population resistance index control)
  - 15% reserve-drug prudence (minimal meropenem use)
  - 15% treatment efficiency  (steps used vs max steps)
"""


class WardManagementGrader:
    """Medium: balance ward outcomes vs population AMR; reward prudent prescribing."""

    def score(self, summary: dict) -> float:
        n = max(1, int(summary.get("n_patients", 1)))
        cured = int(summary.get("cured", 0))
        deaths = int(summary.get("deaths", 0))
        pop = float(summary.get("population_resistance_index", 0.0))
        mero_days = int(summary.get("meropenem_patient_days", 0))
        steps = int(summary.get("steps", 0))
        ward_res = summary.get("ward_res") or {}
        mero_res = float(ward_res.get("meropenem", 0.0))

        # ── Clinical outcomes (40%) ────────────────────────────────────
        cure_rate = cured / n
        death_rate = deaths / n
        clinical = max(0.0, cure_rate - 0.65 * death_rate)

        # ── AMR containment (30%) ──────────────────────────────────────
        # Population resistance index ranges [0, 1]; lower = better stewardship
        amr_containment = max(0.0, 1.0 - 1.2 * pop)

        # ── Reserve-drug prudence (15%) ────────────────────────────────
        # Penalise unnecessary meropenem use; >15 patient-days is excessive
        # for a 5-patient, 32-step medium scenario
        mero_penalty = min(1.0, mero_days / 20.0)
        mero_res_penalty = min(1.0, mero_res / 0.5)  # meropenem resistance spike
        reserve_prudence = max(0.0, 1.0 - 0.6 * mero_penalty - 0.4 * mero_res_penalty)

        # ── Treatment efficiency (15%) ─────────────────────────────────
        max_steps = 32
        if cured >= n and deaths == 0:
            efficiency = max(0.0, 1.0 - (steps / max_steps))
        else:
            efficiency = 0.2 * (cured / n)

        raw = (
            0.40 * clinical
            + 0.30 * amr_containment
            + 0.15 * reserve_prudence
            + 0.15 * efficiency
        )

        # Strictly between 0 and 1 (exclusive) as required by Scaler grader
        return max(0.001, min(0.999, float(raw)))
