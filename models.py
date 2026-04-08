"""
models.py — Typed Pydantic models for Antibiotic Stewardship Env
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class PatientState(BaseModel):
    patient_id: str
    severity_score: float = Field(..., ge=0.0, le=10.0, description="Infection severity (0=healthy, 10=sepsis)")
    infection_cleared: bool = Field(default=False)
    # 0 = susceptible, 1 = resistant
    res_amox: float = Field(..., ge=0.0, le=1.0)
    res_cipro: float = Field(..., ge=0.0, le=1.0)
    res_mero: float = Field(..., ge=0.0, le=1.0)

class WardObservation(BaseModel):
    step_count: int
    max_steps: int
    patients: List[PatientState]
    # Population-level resistance in the ward environment (environmental reservoirs)
    ward_amox_resistance: float = Field(..., ge=0.0, le=1.0)
    ward_cipro_resistance: float = Field(..., ge=0.0, le=1.0)
    ward_mero_resistance: float = Field(..., ge=0.0, le=1.0)
    
    total_patients_cured: int
    total_patient_deaths: int
    budget_remaining: float

class PrescriptionAction(BaseModel):
    patient_id: str
    drug_choice: Literal["none", "amoxicillin", "ciprofloxacin", "meropenem"]
    duration_days: int = Field(default=5, ge=1, le=14)

class StepResult(BaseModel):
    observation: WardObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class ResetResult(BaseModel):
    observation: WardObservation
    info: Dict[str, Any] = Field(default_factory=dict)
