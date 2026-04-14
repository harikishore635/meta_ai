"""
Hard Grader — ReserveBudgetGrader
==================================
Evaluates the agent under a strict meropenem budget and high
baseline resistance. Tests whether the agent can practise true
antimicrobial stewardship under realistic constraints.

Scoring breakdown:
  - 35% clinical outcomes           (cure rate minus death penalty)
  - 25% meropenem budget compliance (staying within patient-day budget)
  - 20% MDR containment             (meropenem + population resistance)
  - 10% narrow-spectrum preference  (low amox/cipro resistance spike)
  - 10% treatment efficiency        (steps used vs max steps)
"""


class ReserveBudgetGrader:
    """Hard: clinical outcomes + strict reserve-drug budget + MDR suppression."""

    def score(self, summary: dict) -> float:
        n = max(1, int(summary.get("n_patients", 1)))
        cured = int(summary.get("cured", 0))
        deaths = int(summary.get("deaths", 0))
        ward_res = summary.get("ward_res") or {}
        mero_res = float(ward_res.get("meropenem", 0.0))
        amox_res = float(ward_res.get("amoxicillin", 0.0))
        cipro_res = float(ward_res.get("ciprofloxacin", 0.0))
        mero_days = int(summary.get("meropenem_patient_days", 0))
        budget = max(1, int(summary.get("meropenem_budget", 14)))
        steps = int(summary.get("steps", 0))
        pop_res = float(summary.get("population_resistance_index", 0.0))

        # ── Clinical outcomes (35%) ────────────────────────────────────
        cure_rate = cured / n
        death_rate = deaths / n
        clinical = max(0.0, cure_rate - 0.70 * death_rate)

        # ── Meropenem budget compliance (25%) ──────────────────────────
        # over-budget patient-days as fraction of budget
        over = max(0.0, float(mero_days - budget))
        if over == 0:
            # Under budget: reward proportional to how much budget was saved
            budget_score = 1.0 - 0.3 * (mero_days / budget)
        else:
            # Over budget: sharp penalty
            budget_score = max(0.0, 1.0 - 0.12 * over)

        # ── MDR containment (20%) ──────────────────────────────────────
        # Combined meropenem resistance + overall population resistance
        mdr = max(0.0, 1.0 - 0.8 * mero_res - 0.5 * pop_res)

        # ── Narrow-spectrum preference (10%) ───────────────────────────
        # If amox/cipro resistance didn't spike much → agent used them wisely
        narrow_avg = (amox_res + cipro_res) / 2.0
        narrow = max(0.0, 1.0 - narrow_avg)

        # ── Treatment efficiency (10%) ─────────────────────────────────
        max_steps = 44
        if cured > 0 and deaths <= n // 3:
            efficiency = max(0.0, 1.0 - (steps / max_steps))
        else:
            efficiency = 0.1 * (cured / n)

        raw = (
            0.35 * clinical
            + 0.25 * budget_score
            + 0.20 * mdr
            + 0.10 * narrow
            + 0.10 * efficiency
        )

        # Strictly between 0 and 1 (exclusive) as required by Scaler grader
        return max(0.001, min(0.999, float(raw)))
