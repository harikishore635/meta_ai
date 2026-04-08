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
This environment simulates a hospital ward where an AI Agent must manage antibiotic prescriptions (Patient-Level dynamics) while considering the long-term evolutionary pressure on bacteria (Population-Level Resistance Dynamics).

Unlike simple single-patient simulation environments, this coupled system models how excessive use of broad-spectrum / reserve antibiotics (like Meropenem) exerts evolutionary pressure on the hospital environment (simulating a Wright-Fisher model of allele selection). 

**The goal is a multi-timescale optimization**: clear the patients' infections fast (hours) without triggering a multi-drug resistant (MDR) outbreak over time (weeks).

## Tasks
1. **Easy (`easy`)**: Single-patient task. Evaluate simple susceptibility profile and prescribe the right narrow-spectrum drug. Grader: Checks success rate of clearing the infection.
2. **Medium (`medium`)**: 5 Patients in a ward with background resistance climbing. Agent must balance cures vs avoiding creating an outbreak.
3. **Hard (`hard`)**: Suppress MDR outbreak with a restrictive budget on the Reserve antibiotic (Meropenem), heavily penalizing any spikes in Meropenem resistance.

## Getting Started

### Run the Server
```bash
python -m uvicorn server:app --port 8000
```

### Try the Inference Script
```bash
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

## Structure
- `models.py`: Strongly-typed Pydantic definitions (Observation, Action)
- `env.py`: The core environment logic (PK/PD model, Resistance evolution, reward computation)
- `graders/`: Deterministic agent graders
- `server.py`: FastAPI server complying with OpenEnv specification.
- `inference.py`: Baseline inference script using Dummy/OpenAI logic.
