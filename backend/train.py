import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.gnn import TransactionGraphGNN, build_graph_data
from backend.models import TransactionAutoencoder, TransactionLSTM, transaction_sequences

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


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = np.random.choice([f"user_{i}" for i in range(USER_COUNT)], size=len(df))
    df["account_id"] = df["user_id"].apply(lambda u: f"acct_{u.split("_")[1]}")
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


def train_autoencoder(X_train: np.ndarray):
    model = TransactionAutoencoder(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    tensor_x = torch.tensor(X_train, dtype=torch.float32)
    for epoch in range(12):
        model.train()
        optimizer.zero_grad()
        recon = model(tensor_x)
        loss = criterion(recon, tensor_x)
        loss.backward()
        optimizer.step()
        if epoch % 4 == 0:
            print(f"Autoencoder epoch {epoch} loss={loss.item():.6f}")
    return model


def train_lstm(X_seq: torch.Tensor, y_seq: torch.Tensor):
    model = TransactionLSTM(input_dim=X_seq.shape[-1])
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    for epoch in range(6):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        print(f"LSTM epoch {epoch} avg_loss={total_loss / len(dataset):.6f}")
    return model


def train_gnn(graph_data, labels):
    model = TransactionGraphGNN(
        node_dim=graph_data["node_features"].shape[1],
        edge_dim=graph_data["edge_features"].shape[1],
        hidden_dim=64,
    )
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.BCELoss()

    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
    edge_feats = torch.tensor(graph_data["edge_features"], dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    sample_size = min(len(pos_idx) * 3, len(neg_idx))
    sampled_neg = np.random.choice(neg_idx, size=sample_size, replace=False)
    train_idx = np.concatenate([pos_idx, sampled_neg])

    for epoch in range(8):
        np.random.shuffle(train_idx)
        model.train()
        epoch_loss = 0.0
        batch_size = 4096
        for start in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[start : start + batch_size]
            optimizer.zero_grad()
            batch_edge_index = edge_index[:, batch_idx]
            batch_edge_feats = edge_feats[batch_idx]
            predictions = model(
                torch.tensor(graph_data["node_features"], dtype=torch.float32),
                batch_edge_index,
                batch_edge_feats,
            )
            loss = criterion(predictions, labels_tensor[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        print(f"GNN epoch {epoch} avg_loss={epoch_loss / len(train_idx):.6f}")
    return model


def main():
    seed_everything()
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
    with open(MODEL_DIR / "scaler.pkl", "wb") as fp:
        pickle.dump(scaler, fp)

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["Class"])
    X_train_legit = scaler.transform(train_df[train_df["Class"] == 0][features].values.astype("float32"))

    print("Training autoencoder on legit transactions...")
    autoencoder = train_autoencoder(X_train_legit)
    torch.save(autoencoder.state_dict(), MODEL_DIR / "autoencoder.pt")

    seq_df = train_df.sort_values("Time")
    X_seq, y_seq = transaction_sequences(seq_df, features, seq_len=8)
    if X_seq is None:
        raise RuntimeError("Not enough sequence data to train LSTM")
    print("Training LSTM on user transaction sequences...")
    lstm = train_lstm(X_seq, y_seq)
    torch.save(lstm.state_dict(), MODEL_DIR / "lstm.pt")

    print("Building graph metadata...")
    graph_data = build_graph_data(train_df)
    torch.save(graph_data, DATA_DIR / "graph_metadata.pt")
    print("Training GNN on transaction graph edges...")
    gnn = train_gnn(graph_data, train_df["Class"].astype("float32").values)
    torch.save(gnn.state_dict(), MODEL_DIR / "gnn.pt")

    print("Training complete. Saved models in models/ and artifacts in data/")


if __name__ == "__main__":
    main()
