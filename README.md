---
title: Antibiotic Stewardship Env
emoji: 🦠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Antibiotic Stewardship Resistance Environment 🦠💊

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-1D9E75)](https://huggingface.co/openenv)

## Overview
This environment simulates a hospital ward where an AI agent must manage antibiotic prescriptions (**patient-level dynamics**) while considering the long-term evolutionary pressure on bacteria (**population-level resistance dynamics**).

Unlike simple single-patient simulation environments, this coupled system models how excessive use of broad-spectrum / reserve antibiotics (like Meropenem) exerts selection pressure on the hospital environment, driving resistance through environmental reservoir drift and treatment-induced collateral selection.

**The goal is a multi-timescale optimization**: clear the patients' infections fast (hours) without triggering a multi-drug resistant (MDR) outbreak over time (weeks).

## Key Design Features

- **Coupled Patient–Ward Dynamics**: Each treatment affects not just the treated patient but drives ward-level resistance via collateral selection pressure. Ward resistance in turn diffuses back to all patients (nosocomial transmission).
- **PK/PD-Inspired Efficacy Model**: Drug efficacy depends on a weighted combination of patient-level and ward-level resistance (`0.62 × patient_res + 0.38 × ward_res`), reflecting both individual isolate susceptibility and environmental resistance pressure.
- **Multi-Component Reward Shaping**: Reward balances cure bonuses (+0.32), death penalties (−0.42), severity reduction, ward resistance drift penalties, and meropenem-specific usage penalties.
- **Reserve-Drug Budget**: In hard mode, meropenem usage is constrained to ≤14 patient-days, modelling real-world formulary restrictions on last-resort antibiotics.

## Tasks
1. **Easy (`easy`)**: Single-patient task (12 steps). Evaluate susceptibility profile and prescribe the right narrow-spectrum drug. Grader: clinical outcome + speed + stewardship.
2. **Medium (`medium`)**: 5 patients in a ward (32 steps) with background resistance climbing. Agent must balance cures vs. avoiding an outbreak. Grader: clinical + AMR containment + reserve-drug prudence + efficiency.
3. **Hard (`hard`)**: 10 patients (44 steps). Suppress MDR outbreak with a strict meropenem patient-day budget (≤14 days). Grader: clinical + budget compliance + MDR containment + narrow-spectrum preference + efficiency.

## Getting Started

### Run the Server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Try the Inference Script
```bash
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

### Run Tests
```bash
pytest tests/ -v
```

## Project Structure
```
├── models.py              # Typed Pydantic models (WardObservation, PrescriptionAction, etc.)
├── env.py                 # Core environment: PK/PD model, resistance dynamics, reward shaping
├── graders/               # Deterministic task graders with multi-component scoring
│   ├── __init__.py
│   ├── easy_grader.py     # SinglePatientGrader  (clinical + speed + stewardship)
│   ├── medium_grader.py   # WardManagementGrader (clinical + AMR + prudence + efficiency)
│   └── hard_grader.py     # ReserveBudgetGrader  (clinical + budget + MDR + narrow + efficiency)
├── server/
│   ├── __init__.py
│   └── app.py             # FastAPI server (/reset, /step, /state, /health)
├── inference.py           # Inference script (LLM + heuristic fallback)
├── client.py              # OpenEnv client wrapper
├── tests/
│   ├── __init__.py
│   └── test_env.py        # 20+ unit & integration tests
├── openenv.yaml           # OpenEnv specification
├── Dockerfile             # Docker deployment (HF Spaces compatible)
├── pyproject.toml          # Package configuration
└── requirements.txt       # Python dependencies
```

## Environment Details

### Observation Space (`WardObservation`)
| Field | Type | Description |
|-------|------|-------------|
| `patients` | `List[PatientState]` | Per-patient severity, cleared status, resistance per drug |
| `ward_{amox,cipro,mero}_resistance` | `float [0,1]` | Ward-level environmental resistance reservoir |
| `population_resistance_index` | `float` | Mean ward resistance across all drugs |
| `meropenem_cumulative_patient_days` | `int` | Total meropenem usage so far |
| `meropenem_patient_day_budget` | `int` | Budget cap (hard mode = 14) |
| `step_count / max_steps` | `int` | Episode progress |

### Action Space (`PrescriptionAction`)
| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | `str` | Target patient (e.g., `"P0"`) |
| `drug_choice` | `Literal` | `"none"`, `"amoxicillin"`, `"ciprofloxacin"`, `"meropenem"` |
| `duration_days` | `int [1,14]` | Treatment duration in days |

## Pharmacological Constants Reference
The PK/PD model uses empirically calibrated constants derived from WHO AMR surveillance guidelines and published antibiotic stewardship literature:

| Constant | Value | Rationale |
|----------|-------|-----------|
| Patient/ward resistance blend | 62% / 38% | Reflects that individual isolate MIC dominates but ward ecology contributes |
| Efficacy threshold | 0.44–0.50 | Minimum efficacy/severity ratio for treatment success (scales with difficulty) |
| Ward drift per step | 0.003–0.010 | Background resistance accumulation from environmental reservoirs |
| Collateral selection (meropenem) | 0.0035/day | Cross-resistance pressure on amox/cipro from carbapenem use |
| Patient–ward sync rate (α) | 0.08–0.12 | Nosocomial transmission rate of resistant organisms |
