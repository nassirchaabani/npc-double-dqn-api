import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field

from .model import encode_observation, load_model

app = FastAPI(title="NPC DQN API", version="0.1.0")
PREDICTIONS = Counter("npc_predictions_total", "Number of NPC action predictions")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/npc_dqn.pt"))
model = None


class Observation(BaseModel):
    values: list[float] = Field(min_length=6, max_length=6)


@app.on_event("startup")
def initialize_model():
    global model
    model = load_model(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(observation: Observation):
    if model is None:
        raise HTTPException(status_code=503, detail="No trained model is available")
    with torch.no_grad():
        encoded = encode_observation(observation.values)
        q_values = model(torch.tensor(encoded, dtype=torch.float32).unsqueeze(0))[0]
    PREDICTIONS.inc()
    return {"action": int(q_values.argmax().item()), "q_values": q_values.tolist()}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
