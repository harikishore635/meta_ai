class SinglePatientGrader:
    """Easy: clear infection without death; partial credit for survival + low severity."""

    def score(self, summary: dict) -> float:
        if summary.get("deaths", 0) > 0:
            raw = 0.0
        elif summary.get("cured", 0) >= 1:
            raw = 1.0
        else:
            mean_sev = float(summary.get("mean_severity_living", 10.0))
            raw = (10.0 - mean_sev) / 10.0 * 0.35
        # Strictly between 0 and 1 (exclusive) as required by Scaler grader
        return max(0.001, min(0.999, float(raw)))
