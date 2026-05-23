import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.explain import explain_transaction
from backend.gnn import TransactionGraphGNN
from backend.models import TransactionAutoencoder, TransactionLSTM

LIVE_FEED_COUNT = 8
MERCHANT_NAMES = [
    "Amazon",
    "Flipkart",
    "Paytm",
    "Swiggy",
    "Myntra",
    "Zomato",
    "Uber Eats",
    "IRCTC",
    "Netflix",
    "Spotify",
]


def format_seconds_to_time(seconds: int) -> str:
    hours = (seconds // 3600) % 24
    minutes = (seconds % 3600) // 60
    return f"{hours:02}:{minutes:02}"


def sample_live_transactions(count: int = LIVE_FEED_COUNT):
    rows = []
    source_path = DATA_DIR / "processed.parquet"
    if source_path.exists():
        df = pd.read_parquet(source_path)
    elif RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
    else:
        return []

    if df.empty:
        return []

    sample = df.sample(n=min(count, len(df)), replace=False, random_state=random.randint(1, 100000))
    for _, row in sample.iterrows():
        time_value = int(row.Time) if "Time" in row and not pd.isna(row.Time) else random.randint(0, 86399)
        amount = float(row.Amount) if "Amount" in row and not pd.isna(row.Amount) else float(random.randint(100, 90000))
        verdict = "FRAUD" if "Class" in row and int(row.Class) == 1 else "LEGIT"
        location = row.location if "location" in row and pd.notna(row.location) else random.choice(CITY_POOL)
        device_id = row.device_id if "device_id" in row and pd.notna(row.device_id) else random.choice(DEVICE_POOL)
        user_id = row.user_id if "user_id" in row and pd.notna(row.user_id) else f"user_{int(_ % USER_COUNT)}"
        merchant = random.choice(MERCHANT_NAMES)
        rows.append(
            {
                "user_id": str(user_id),
                "amount": round(amount, 2),
                "time": format_seconds_to_time(time_value),
                "location": location,
                "device_id": device_id,
                "merchant": merchant,
                "verdict": verdict,
            }
        )

    return rows

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

    graph_metadata = None
    graph_path = DATA_DIR / "graph_metadata.pt"
    if graph_path.exists():
        try:
            with open(graph_path, "rb") as fp:
                graph_metadata = pickle.load(fp)
        except Exception:
            graph_metadata = None

    autoencoder_model = None
    lstm_model = None
    gnn_model = None
    try:
        autoencoder_path = MODEL_DIR / "autoencoder.pth"
        if autoencoder_path.exists():
            autoencoder_model = TransactionAutoencoder(3)
            autoencoder_model.load_state_dict(torch.load(autoencoder_path, map_location="cpu"))
            autoencoder_model.eval()
    except Exception:
        autoencoder_model = None

    try:
        lstm_path = MODEL_DIR / "lstm.pth"
        if lstm_path.exists():
            lstm_model = TransactionLSTM(3)
            lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
            lstm_model.eval()
    except Exception:
        lstm_model = None

    try:
        gnn_path = MODEL_DIR / "gnn.pth"
        if gnn_path.exists():
            edge_dim = graph_metadata["edge_feature_dim"] if graph_metadata else 3
            gnn_model = TransactionGraphGNN(3, edge_dim)
            gnn_model.load_state_dict(torch.load(gnn_path, map_location="cpu"))
            gnn_model.eval()
    except Exception:
        gnn_model = None

    return {
        "scaler": scaler,
        "profiles": profiles,
        "graph_metadata": graph_metadata,
        "autoencoder_model": autoencoder_model,
        "lstm_model": lstm_model,
        "gnn_model": gnn_model,
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
    return artifacts["scaler"].transform(x).flatten()


def get_profile(user_id: str):
    return artifacts["profiles"].get(user_id, {
        "avg_amount": 0.0,
        "locations": [],
        "known_devices": [],
        "recent_sequence": [],
    })


def score_with_autoencoder(feature_vec: np.ndarray) -> float:
    model = artifacts.get("autoencoder_model")
    if model is None or feature_vec is None:
        anomaly = float(np.mean(np.abs(feature_vec))) if feature_vec is not None else 0.25
        score = 1.0 / (1.0 + np.exp(-np.clip((anomaly - 0.4) * 3.0, -10.0, 10.0)))
        return float(np.clip(score, 0.02, 0.98))

    with torch.no_grad():
        x = torch.tensor(feature_vec.reshape(1, -1), dtype=torch.float32)
        recon = model(x)
        error = torch.mean(torch.abs(x - recon)).item()
    score = 1.0 / (1.0 + np.exp(-np.clip((error - 0.04) * 20.0, -10.0, 10.0)))
    return float(np.clip(score, 0.02, 0.98))


def score_with_lstm(transaction: Dict, profile: Dict) -> float:
    model = artifacts.get("lstm_model")
    sequence = profile.get("recent_sequence", [])
    if model is not None and len(sequence) >= 8:
        seq_array = np.array(sequence[-8:], dtype="float32")
        seq_tensor = torch.tensor(seq_array.reshape(1, 8, 3), dtype=torch.float32)
        with torch.no_grad():
            score = float(model(seq_tensor).item())
        return float(np.clip(score, 0.02, 0.98))

    if profile.get("avg_amount", 0.0) > 0 and transaction["Amount"] > profile["avg_amount"] * 5:
        return 0.6
    if transaction["Amount"] >= 100000:
        return 0.72
    return 0.15


def score_with_gnn(transaction: Dict) -> float:
    metadata = artifacts.get("graph_metadata")
    model = artifacts.get("gnn_model")
    if model is None or not metadata:
        return 0.2
    node_index = metadata.get("node_index", {})
    source = node_index.get(transaction["account_id"])
    target = node_index.get(transaction["merchant_id"])
    if source is None or target is None:
        return 0.2
    edge_feats = np.array([
        [transaction["Amount"], transaction["hour_of_day"], transaction["is_night"]]
    ], dtype="float32")
    edge_scaler = metadata.get("edge_scaler")
    if edge_scaler is not None:
        try:
            edge_feats = edge_scaler.transform(edge_feats)
        except Exception:
            pass
    node_features = np.array(metadata.get("node_features", []), dtype="float32")
    if node_features.size == 0 or source >= len(node_features) or target >= len(node_features):
        return 0.2

    node_tensor = torch.tensor(node_features, dtype=torch.float32)
    edge_tensor = torch.tensor(edge_feats, dtype=torch.float32)
    edge_index = torch.tensor([[source], [target]], dtype=torch.long)
    with torch.no_grad():
        score = float(model(node_tensor, edge_index, edge_tensor).item())
    return float(np.clip(score, 0.02, 0.98))


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


@app.get("/live-transactions")
def live_transactions(count: int = LIVE_FEED_COUNT):
    try:
        return sample_live_transactions(count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": models_loaded}
