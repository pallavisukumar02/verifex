import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TransactionGraphGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.node_encoder(node_features))
        edge_h = torch.relu(self.edge_encoder(edge_features))
        source, target = edge_index
        src_h = h[source]
        tgt_h = h[target]
        combined = torch.cat([src_h, tgt_h, edge_h], dim=-1)
        message = self.message(combined)
        return self.edge_classifier(message).squeeze(-1)


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


def train_gnn(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_features: torch.Tensor,
    edge_labels: torch.Tensor,
    hidden_dim: int = 64,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TransactionGraphGNN:
    device = torch.device(device)
    model = TransactionGraphGNN(node_features.size(1), edge_features.size(1), hidden_dim=hidden_dim).to(device)
    dataset = TensorDataset(edge_index.t().float(), edge_features, edge_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for edge_batch, feat_batch, target_batch in loader:
            edge_batch = edge_batch.long().t().to(device)
            feat_batch = feat_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            outputs = model(node_features.to(device), edge_batch, feat_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target_batch.size(0)
            total_count += target_batch.size(0)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            avg_loss = total_loss / max(total_count, 1)
            print(f"GNN epoch {epoch}/{epochs} loss={avg_loss:.6f}")

    return model
