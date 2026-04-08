class SinglePatientGrader:
    """Easy: clear infection without death; partial credit for survival + low severity."""

    def score(self, summary: dict) -> float:
        if summary.get("deaths", 0) > 0:
            return 0.0
        if summary.get("cured", 0) >= 1:
            return 1.0
        mean_sev = float(summary.get("mean_severity_living", 10.0))
        return max(0.0, min(1.0, (10.0 - mean_sev) / 10.0 * 0.35))
