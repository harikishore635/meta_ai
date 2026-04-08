import os
import sys

# Ensure root directory is reliably in path for Docker imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from env import AntibioticEnv
from models import PrescriptionAction, ResetResult, StepResult

env_instance = AntibioticEnv()

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int | None = None

app = FastAPI(title="Antibiotic Stewardship Env", version="0.1.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/reset", response_model=ResetResult)
def reset_env(req: ResetRequest | None = None):
    req = req or ResetRequest()
    return env_instance.reset(task_id=req.task_id, seed=req.seed)

@app.post("/step", response_model=StepResult)
def step_env(action: PrescriptionAction):
    return env_instance.step(action)

@app.get("/state")
def get_state():
    return env_instance.state()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
