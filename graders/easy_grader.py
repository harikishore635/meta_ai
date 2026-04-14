"""
Easy Grader — SinglePatientGrader
=================================
Evaluates whether the agent can clear a single patient's infection
using appropriate (preferably narrow-spectrum) antibiotic therapy.

Scoring breakdown:
  - 60% clinical outcome  (cure = full, death = 0, else partial by severity)
  - 20% treatment speed   (fewer steps → higher bonus)
  - 20% resistance impact (lower final ward resistance → higher bonus)
"""


class SinglePatientGrader:
    """Easy: clear infection with appropriate therapy; penalise overkill."""

    def score(self, summary: dict) -> float:
        deaths = int(summary.get("deaths", 0))
        cured = int(summary.get("cured", 0))
        mean_sev = float(summary.get("mean_severity_living", 10.0))
        steps = int(summary.get("steps", 0))
        pop_res = float(summary.get("population_resistance_index", 0.0))

        # ── Clinical outcome (60%) ─────────────────────────────────────
        if deaths > 0:
            clinical = 0.0
        elif cured >= 1:
            clinical = 1.0
        else:
            # Partial credit: how close to clearing?
            clinical = max(0.0, (10.0 - mean_sev) / 10.0) * 0.45

        # ── Treatment speed (20%) ──────────────────────────────────────
        # Max 12 steps for easy; faster = better
        max_steps = 12
        speed = max(0.0, 1.0 - (steps / max_steps)) if cured >= 1 else 0.0

        # ── Resistance stewardship (20%) ────────────────────────────────
        # Lower population resistance index at end → better stewardship
        stewardship = max(0.0, 1.0 - pop_res)

        raw = 0.60 * clinical + 0.20 * speed + 0.20 * stewardship

        # Strictly between 0 and 1 (exclusive) as required by Scaler grader
        return max(0.001, min(0.999, float(raw)))
