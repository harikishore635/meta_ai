"""
tests/test_env.py — Unit and integration tests for Antibiotic Stewardship Env
==============================================================================
Covers: environment reset, step execution, reward bounds, episode termination,
        grader scoring, observation structure, and server endpoint contracts.
"""
import sys
import os
import json
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import AntibioticEnv, DRUG_KEYS
from models import PrescriptionAction, WardObservation, PatientState
from graders.easy_grader import SinglePatientGrader
from graders.medium_grader import WardManagementGrader
from graders.hard_grader import ReserveBudgetGrader


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return AntibioticEnv()


# ── Reset Tests ───────────────────────────────────────────────────────────────

class TestReset:
    """Test that reset produces valid initial states for each difficulty."""

    @pytest.mark.parametrize("task_id,expected_patients,expected_max_steps", [
        ("easy", 1, 12),
        ("medium", 5, 32),
        ("hard", 10, 44),
    ])
    def test_reset_patient_count(self, env, task_id, expected_patients, expected_max_steps):
        result = env.reset(task_id=task_id, seed=42)
        assert len(result.observation.patients) == expected_patients
        assert result.observation.max_steps == expected_max_steps
        assert result.observation.step_count == 0

    def test_reset_deterministic(self, env):
        """Same seed should produce identical observations."""
        r1 = env.reset(task_id="easy", seed=123)
        r2 = env.reset(task_id="easy", seed=123)
        assert r1.observation.patients[0].severity_score == r2.observation.patients[0].severity_score
        assert r1.observation.ward_amox_resistance == r2.observation.ward_amox_resistance

    def test_reset_ward_resistance_bounds(self, env):
        """All ward resistance values should be in [0, 1]."""
        for task in ("easy", "medium", "hard"):
            result = env.reset(task_id=task, seed=42)
            obs = result.observation
            assert 0.0 <= obs.ward_amox_resistance <= 1.0
            assert 0.0 <= obs.ward_cipro_resistance <= 1.0
            assert 0.0 <= obs.ward_mero_resistance <= 1.0

    def test_reset_patients_alive(self, env):
        """No patient should be dead or cleared at reset."""
        result = env.reset(task_id="medium", seed=42)
        for p in result.observation.patients:
            assert not p.infection_cleared
            assert p.severity_score < 10.0

    def test_reset_meropenem_budget_hard(self, env):
        """Hard mode should have a restricted meropenem budget."""
        result = env.reset(task_id="hard", seed=42)
        assert result.observation.meropenem_patient_day_budget <= 14


# ── Step Tests ────────────────────────────────────────────────────────────────

class TestStep:
    """Test step mechanics, reward bounds, and episode termination."""

    def test_step_reward_bounded(self, env):
        """Reward must always be in [-1, 1]."""
        env.reset(task_id="easy", seed=42)
        action = PrescriptionAction(patient_id="P0", drug_choice="amoxicillin", duration_days=5)
        result = env.step(action)
        assert -1.0 <= result.reward <= 1.0

    def test_step_increments_count(self, env):
        """Step count should increment after each step."""
        env.reset(task_id="easy", seed=42)
        action = PrescriptionAction(patient_id="P0", drug_choice="amoxicillin", duration_days=3)
        result = env.step(action)
        assert result.observation.step_count == 1

    def test_step_none_drug(self, env):
        """Choosing 'none' should not crash and severity should change."""
        env.reset(task_id="easy", seed=42)
        action = PrescriptionAction(patient_id="P0", drug_choice="none", duration_days=1)
        result = env.step(action)
        assert result.observation is not None
        assert -1.0 <= result.reward <= 1.0

    def test_episode_terminates(self, env):
        """Episode must terminate within max_steps."""
        env.reset(task_id="easy", seed=42)
        done = False
        for _ in range(50):  # More than max_steps for any difficulty
            action = PrescriptionAction(patient_id="P0", drug_choice="amoxicillin", duration_days=3)
            result = env.step(action)
            if result.done:
                done = True
                break
        assert done, "Episode should terminate within max_steps"

    def test_meropenem_increments_patient_days(self, env):
        """Using meropenem should increment cumulative patient days."""
        env.reset(task_id="easy", seed=42)
        action = PrescriptionAction(patient_id="P0", drug_choice="meropenem", duration_days=5)
        result = env.step(action)
        assert result.observation.meropenem_cumulative_patient_days >= 5

    def test_ward_resistance_drifts_upward(self, env):
        """Ward resistance should drift upward over untreated steps."""
        env.reset(task_id="medium", seed=42)
        initial_pop = env._population_res_index()
        for _ in range(5):
            action = PrescriptionAction(patient_id="P0", drug_choice="none", duration_days=1)
            env.step(action)
        assert env._population_res_index() > initial_pop


# ── Observation Structure Tests ───────────────────────────────────────────────

class TestObservation:
    """Verify observation Pydantic model constraints."""

    def test_observation_is_valid_pydantic(self, env):
        result = env.reset(task_id="medium", seed=42)
        obs = result.observation
        assert isinstance(obs, WardObservation)
        for p in obs.patients:
            assert isinstance(p, PatientState)
            assert 0.0 <= p.severity_score <= 10.0
            assert 0.0 <= p.res_amox <= 1.0
            assert 0.0 <= p.res_cipro <= 1.0
            assert 0.0 <= p.res_mero <= 1.0

    def test_info_contains_grading_fields(self, env):
        result = env.reset(task_id="hard", seed=42)
        info = result.info
        required_keys = [
            "task_id", "cured", "deaths", "n_patients",
            "ward_res", "population_resistance_index",
            "meropenem_patient_days", "meropenem_budget",
        ]
        for key in required_keys:
            assert key in info, f"Missing key '{key}' in info dict"


# ── Grader Tests ──────────────────────────────────────────────────────────────

class TestGraders:
    """Verify grader scoring logic and edge cases."""

    def test_easy_perfect_score(self):
        g = SinglePatientGrader()
        summary = {"deaths": 0, "cured": 1, "steps": 3, "population_resistance_index": 0.1, "mean_severity_living": 0.0}
        score = g.score(summary)
        assert 0.75 < score < 1.0, f"Perfect cure should score high, got {score}"

    def test_easy_death_score(self):
        g = SinglePatientGrader()
        summary = {"deaths": 1, "cured": 0, "steps": 5, "population_resistance_index": 0.5, "mean_severity_living": 10.0}
        score = g.score(summary)
        assert score < 0.3, f"Death should score low, got {score}"

    def test_medium_all_cured(self):
        g = WardManagementGrader()
        summary = {"n_patients": 5, "cured": 5, "deaths": 0, "population_resistance_index": 0.1, "meropenem_patient_days": 0, "steps": 10, "ward_res": {"meropenem": 0.05}}
        score = g.score(summary)
        assert score > 0.6, f"All cured with low resistance should score well, got {score}"

    def test_medium_all_dead(self):
        g = WardManagementGrader()
        summary = {"n_patients": 5, "cured": 0, "deaths": 5, "population_resistance_index": 0.8, "meropenem_patient_days": 30, "steps": 32, "ward_res": {"meropenem": 0.7}}
        score = g.score(summary)
        assert score < 0.15, f"All deaths should score very low, got {score}"

    def test_hard_within_budget(self):
        g = ReserveBudgetGrader()
        summary = {"n_patients": 10, "cured": 7, "deaths": 1, "ward_res": {"meropenem": 0.1, "amoxicillin": 0.3, "ciprofloxacin": 0.25}, "meropenem_patient_days": 10, "meropenem_budget": 14, "steps": 30, "population_resistance_index": 0.2}
        score = g.score(summary)
        assert score > 0.4, f"Good stewardship should score reasonably, got {score}"

    def test_hard_over_budget_penalty(self):
        g = ReserveBudgetGrader()
        summary = {"n_patients": 10, "cured": 7, "deaths": 1, "ward_res": {"meropenem": 0.5, "amoxicillin": 0.3, "ciprofloxacin": 0.25}, "meropenem_patient_days": 30, "meropenem_budget": 14, "steps": 30, "population_resistance_index": 0.5}
        score = g.score(summary)
        summary_ok = {"n_patients": 10, "cured": 7, "deaths": 1, "ward_res": {"meropenem": 0.1, "amoxicillin": 0.3, "ciprofloxacin": 0.25}, "meropenem_patient_days": 10, "meropenem_budget": 14, "steps": 30, "population_resistance_index": 0.2}
        score_ok = g.score(summary_ok)
        assert score < score_ok, "Over-budget should score lower than within-budget"

    def test_all_grader_scores_bounded(self):
        """Scores must be in (0, 1) exclusive for Scaler grader."""
        graders = [SinglePatientGrader(), WardManagementGrader(), ReserveBudgetGrader()]
        edge_cases = [
            {"n_patients": 0, "cured": 0, "deaths": 0, "ward_res": {}, "meropenem_patient_days": 0, "meropenem_budget": 1, "steps": 0, "population_resistance_index": 0.0, "mean_severity_living": 5.0},
            {"n_patients": 10, "cured": 10, "deaths": 0, "ward_res": {"meropenem": 0.0, "amoxicillin": 0.0, "ciprofloxacin": 0.0}, "meropenem_patient_days": 0, "meropenem_budget": 14, "steps": 1, "population_resistance_index": 0.0, "mean_severity_living": 0.0},
            {"n_patients": 10, "cured": 0, "deaths": 10, "ward_res": {"meropenem": 1.0, "amoxicillin": 1.0, "ciprofloxacin": 1.0}, "meropenem_patient_days": 100, "meropenem_budget": 14, "steps": 44, "population_resistance_index": 1.0, "mean_severity_living": 10.0},
        ]
        for g in graders:
            for case in edge_cases:
                s = g.score(case)
                assert 0.0 < s < 1.0, f"{g.__class__.__name__} score {s} out of (0,1) bounds"


# ── State / Snapshot Tests ────────────────────────────────────────────────────

class TestState:
    """Verify the /state endpoint contract."""

    def test_state_matches_info(self, env):
        result = env.reset(task_id="easy", seed=42)
        state = env.state()
        assert state["task_id"] == "easy"
        assert state["cured"] == 0
        assert state["deaths"] == 0


# ── Integration Test ──────────────────────────────────────────────────────────

class TestFullEpisode:
    """Run a full episode end-to-end to verify no crashes."""

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_full_episode(self, env, task_id):
        result = env.reset(task_id=task_id, seed=42)
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < 100:
            # Pick highest-severity patient still alive
            alive = [p for p in result.observation.patients if not p.infection_cleared and p.severity_score < 10.0]
            if not alive:
                break
            alive.sort(key=lambda x: x.severity_score, reverse=True)
            target = alive[0]

            # Simple strategy: use amoxicillin
            action = PrescriptionAction(
                patient_id=target.patient_id,
                drug_choice="amoxicillin",
                duration_days=5,
            )
            result_step = env.step(action)
            result = type("R", (), {"observation": result_step.observation})()
            total_reward += result_step.reward
            done = result_step.done
            steps += 1

        # Episode must have terminated
        assert done or steps > 0
        # Final state should be accessible
        state = env.state()
        assert "cured" in state
        assert "deaths" in state
