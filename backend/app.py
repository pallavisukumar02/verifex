import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.explain import explain_transaction
from backend.gnn import TransactionGraphGNN
from backend.models import TransactionAutoencoder, TransactionLSTM

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parent / "models"
DATA_DIR = ROOT.parent / "data"

app = FastAPI(title="Verifex Fraud Scoring API")

# Allow CORS from local frontend dev server(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionRequest(BaseModel):
    user_id: str
    amount: float
    location: str
    device_id: str
    time: str = Field(..., description="HH:MM timestamp for the transaction")
    account_id: Optional[str] = None
    merchant_id: Optional[str] = None


class ScoreResponse(BaseModel):
    risk_score: float
    verdict: str
    reasons: List[str]
    contributions: Dict[str, float]


def load_artifacts():
    with open(MODEL_DIR / "scaler.pkl", "rb") as fp:
        scaler = pickle.load(fp)
    profiles = json.loads((DATA_DIR / "user_profiles.json").read_text())
    graph_metadata = torch.load(DATA_DIR / "graph_metadata.pt", map_location="cpu", weights_only=False)

    autoencoder = TransactionAutoencoder(input_dim=scaler.scale_.shape[0])
    autoencoder.load_state_dict(torch.load(MODEL_DIR / "autoencoder.pt", map_location="cpu", weights_only=False))
    autoencoder.eval()

    lstm = TransactionLSTM(input_dim=3)
    lstm.load_state_dict(torch.load(MODEL_DIR / "lstm.pt", map_location="cpu", weights_only=False))
    lstm.eval()

    gnn = TransactionGraphGNN(
        node_dim=graph_metadata["node_feature_dim"],
        edge_dim=graph_metadata["edge_feature_dim"],
        hidden_dim=64,
    )
    gnn.load_state_dict(torch.load(MODEL_DIR / "gnn.pt", map_location="cpu", weights_only=False))
    gnn.eval()

    return {
        "scaler": scaler,
        "profiles": profiles,
        "graph_metadata": graph_metadata,
        "autoencoder": autoencoder,
        "lstm": lstm,
        "gnn": gnn,
        "node_features": torch.tensor(np.array(graph_metadata["node_features"]), dtype=torch.float32),
    }


artifacts = None
models_loaded = False
try:
    artifacts = load_artifacts()
    models_loaded = True
except Exception as exc:
    print(f"Warning: model artifacts unavailable: {exc}")


def parse_time_to_seconds(ts: str) -> int:
    parts = ts.split(":")
    if len(parts) != 2:
        raise ValueError("time must be HH:MM")
    return int(parts[0]) * 3600 + int(parts[1]) * 60


def build_transaction(transaction: TransactionRequest) -> Dict:
    hour = parse_time_to_seconds(transaction.time) // 3600
    return {
        "Amount": float(transaction.amount),
        "location": transaction.location,
        "device_id": transaction.device_id,
        "hour_of_day": int(hour),
        "is_night": 1.0 if hour < 6 or hour >= 23 else 0.0,
        "account_id": transaction.account_id or f"acct_{transaction.user_id.split('_')[-1]}",
        "merchant_id": transaction.merchant_id or "merchant_0",
    }


def feature_vector(transaction: Dict):
    raw = [transaction["Amount"], transaction["hour_of_day"], transaction["is_night"]]
    x = np.array(raw, dtype="float32").reshape(1, -1)
    return torch.tensor(artifacts["scaler"].transform(x), dtype=torch.float32)


def get_profile(user_id: str):
    return artifacts["profiles"].get(user_id, {
        "avg_amount": 0.0,
        "locations": [],
        "known_devices": [],
        "recent_sequence": [],
    })


def score_with_autoencoder(feature_vec: torch.Tensor) -> float:
    with torch.no_grad():
        recon = artifacts["autoencoder"](feature_vec)
        mse = torch.mean((recon - feature_vec) ** 2).item()
        return float(torch.sigmoid(torch.tensor(mse * 10)).item())


def score_with_lstm(transaction: Dict, profile: Dict) -> float:
    recent = profile.get("recent_sequence", [])
    if len(recent) >= 8:
        seq = torch.tensor([recent], dtype=torch.float32)
        with torch.no_grad():
            return float(artifacts["lstm"](seq).item())
    if profile.get("avg_amount", 0.0) > 0 and transaction["Amount"] > profile["avg_amount"] * 5:
        return 0.6
    if transaction["Amount"] >= 100000:
        return 0.72
    return 0.15


def score_with_gnn(transaction: Dict) -> float:
    metadata = artifacts["graph_metadata"]
    node_index = metadata["node_index"]
    source = node_index.get(transaction["account_id"])
    target = node_index.get(transaction["merchant_id"])
    if source is None or target is None:
        return 0.2
    edge_feats = np.array([
        [transaction["Amount"], transaction["hour_of_day"], transaction["is_night"]]
    ], dtype="float32")
    edge_feats = metadata["edge_scaler"].transform(edge_feats)
    edge_index = torch.tensor([[source], [target]], dtype=torch.long)
    with torch.no_grad():
        score = artifacts["gnn"](artifacts["node_features"], edge_index, torch.tensor(edge_feats, dtype=torch.float32))
        return float(score.item())


@app.post("/score-transaction", response_model=ScoreResponse)
def score_transaction(payload: TransactionRequest):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded. Train models first.")
    tx = build_transaction(payload)
    profile = get_profile(payload.user_id)
    feature_vec = feature_vector(tx)
    auto_score = score_with_autoencoder(feature_vec)
    lstm_score = score_with_lstm(tx, profile)
    gnn_score = score_with_gnn(tx)
    return explain_transaction(tx, profile, auto_score, lstm_score, gnn_score)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": models_loaded}
