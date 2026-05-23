import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from backend.gnn import train_gnn
from backend.models import train_autoencoder, train_lstm, transaction_sequences

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
MODEL_DIR = ROOT.parent / "models"
RAW_PATH = ROOT.parent / "creditcard.csv"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CITY_POOL = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"]
DEVICE_POOL = ["phone_1", "phone_2", "tablet_1", "laptop_1", "phone_3", "phone_4"]
MERCHANT_COUNT = 1200
USER_COUNT = 2500


def augment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = np.random.choice([f"user_{i}" for i in range(USER_COUNT)], size=len(df))
    df["account_id"] = df["user_id"].apply(lambda u: f"acct_{u.split('_')[1]}")
    df["merchant_id"] = np.random.choice([f"merchant_{i}" for i in range(MERCHANT_COUNT)], size=len(df))

    location_map = {}
    device_map = {}
    for user in df["user_id"].unique():
        primary = np.random.choice(CITY_POOL)
        secondaries = list(np.random.choice([c for c in CITY_POOL if c != primary], size=2, replace=False))
        location_map[user] = [primary] + secondaries
        device_map[user] = list(np.random.choice(DEVICE_POOL, size=np.random.randint(1, 4), replace=False))

    def sample_location(user):
        choices = location_map[user]
        return np.random.choice(choices, p=[0.7, 0.2, 0.1])

    def sample_device(user):
        return np.random.choice(device_map[user])

    df["location"] = df["user_id"].map(sample_location)
    df["device_id"] = df["user_id"].map(sample_device)
    df["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(int)
    df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 23)).astype(float)
    df["amount_log"] = np.log1p(df["Amount"])
    return df


def build_user_profiles(df: pd.DataFrame):
    profiles = {}
    seq_features = ["Amount", "hour_of_day", "is_night"]
    for user, subset in df.sort_values("Time").groupby("user_id"):
        profiles[user] = {
            "avg_amount": float(subset["Amount"].mean()),
            "locations": sorted(subset["location"].value_counts().index.tolist())[:3],
            "known_devices": sorted(subset["device_id"].unique().tolist()),
            "recent_sequence": subset[seq_features].tail(8).values.astype("float32").tolist(),
        }
    return profiles


def build_graph_data(df: pd.DataFrame):
    nodes = sorted(set(df["account_id"]).union(df["merchant_id"]))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    edge_source = [node_index[a] for a in df["account_id"].tolist()]
    edge_target = [node_index[m] for m in df["merchant_id"].tolist()]
    edge_index = np.vstack([edge_source, edge_target]).astype(np.int64)

    node_features = []
    for node in nodes:
        subset = df[(df["account_id"] == node) | (df["merchant_id"] == node)]
        count = float(len(subset))
        avg_amount = float(subset["Amount"].mean()) if len(subset) else 0.0
        fraud_ratio = float(subset["Class"].mean()) if len(subset) else 0.0
        node_features.append([count, avg_amount, fraud_ratio])
    node_features = np.array(node_features, dtype=np.float32)

    hours = ((df["Time"] // 3600) % 24).astype(np.float32).values
    edge_features = np.stack(
        [df["Amount"].astype(np.float32).values, hours, ((hours < 6) | (hours >= 23)).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    edge_scaler = StandardScaler().fit(edge_features)
    edge_features = edge_scaler.transform(edge_features).astype(np.float32)

    return {
        "node_index": node_index,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_scaler": edge_scaler,
        "node_feature_dim": node_features.shape[1],
        "edge_feature_dim": edge_features.shape[1],
    }


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    print("Loading raw dataset...")
    raw = pd.read_csv(RAW_PATH)
    df = augment_dataset(raw)
    df.to_parquet(DATA_DIR / "processed.parquet", index=False)

    profiles = build_user_profiles(df)
    with open(DATA_DIR / "user_profiles.json", "w", encoding="utf-8") as fp:
        json.dump(profiles, fp, indent=2)

    features = ["Amount", "hour_of_day", "is_night"]
    scaler = StandardScaler().fit(df[features].values.astype("float32"))
    feature_array = scaler.transform(df[features].values.astype("float32")).astype("float32")
    feature_tensor = torch.from_numpy(feature_array)

    print("Training autoencoder...")
    autoencoder = train_autoencoder(feature_tensor, epochs=20, batch_size=1024, lr=1e-3)
    torch.save(autoencoder.state_dict(), MODEL_DIR / "autoencoder.pth")

    print("Building LSTM sequence dataset...")
    sequences, labels = transaction_sequences(df, features, seq_len=8)
    if sequences is not None and labels is not None:
        print(f"Training LSTM on {len(sequences)} sequences...")
        lstm = train_lstm(sequences, labels, hidden_dim=64, num_layers=2, epochs=20, batch_size=512, lr=1e-3)
        torch.save(lstm.state_dict(), MODEL_DIR / "lstm.pth")
    else:
        print("Skipping LSTM training because no sequence data was generated.")

    print("Building graph metadata...")
    graph_data = build_graph_data(df)
    with open(DATA_DIR / "graph_metadata.pt", "wb") as fp:
        pickle.dump(graph_data, fp)

    print("Training graph model...")
    node_features = torch.from_numpy(graph_data["node_features"])
    edge_index = torch.from_numpy(graph_data["edge_index"])
    edge_features = torch.from_numpy(graph_data["edge_features"])
    edge_labels = torch.from_numpy(df["Class"].values.astype("float32")).unsqueeze(1)
    gnn = train_gnn(node_features, edge_index, edge_features, edge_labels, hidden_dim=64, epochs=20, batch_size=1024, lr=1e-3)
    torch.save(gnn.state_dict(), MODEL_DIR / "gnn.pth")

    with open(MODEL_DIR / "scaler.pkl", "wb") as fp:
        pickle.dump(scaler, fp)

    print("Artifact generation complete. Saved scaler, profile, graph metadata, and model weights.")


if __name__ == "__main__":
    main()
