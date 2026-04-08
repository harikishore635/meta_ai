class ReserveBudgetGrader:
    """Hard: clinical outcomes + strict control of reserve (meropenem) use and ward MDR."""

    def score(self, summary: dict) -> float:
        n = max(1, int(summary.get("n_patients", 1)))
        cured = int(summary.get("cured", 0))
        deaths = int(summary.get("deaths", 0))
        w = summary.get("ward_res") or {}
        mero = float(w.get("meropenem", 0.0))
        days = int(summary.get("meropenem_patient_days", 0))
        budget = max(1, int(summary.get("meropenem_budget", 14)))
        over = max(0.0, float(days - budget))

        clinical = (cured / n) - 0.6 * (deaths / n)
        reserve = 0.75 * mero
        budget_pen = 0.04 * over
        score = clinical - reserve - budget_pen
        return max(0.0, min(1.0, float(score)))
