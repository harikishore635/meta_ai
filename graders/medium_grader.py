class WardManagementGrader:
    """Medium: balance outcomes vs population AMR (deterministic, bounded)."""

    def score(self, summary: dict) -> float:
        n = max(1, int(summary.get("n_patients", 1)))
        cured = int(summary.get("cured", 0))
        deaths = int(summary.get("deaths", 0))
        pop = float(summary.get("population_resistance_index", 0.0))

        clinical = (cured / n) - 0.55 * (deaths / n)
        amr_term = 0.55 * pop
        score = clinical - amr_term
        # Strictly between 0 and 1 (exclusive) as required by Scaler grader
        return max(0.001, min(0.999, float(score)))
