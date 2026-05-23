import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TransactionAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            prev = dim
        self.encoder = nn.Sequential(*layers)

        decoder = []
        for dim in reversed(hidden_dims[:-1]):
            decoder.append(nn.Linear(prev, dim))
            decoder.append(nn.ReLU())
            prev = dim
        decoder.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)


class TransactionLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        final_hidden = outputs[:, -1, :]
        return self.classifier(final_hidden)


def transaction_sequences(df, feature_columns, seq_len=8):
    sequences = []
    labels = []
    for _, group in df.sort_values("Time").groupby("user_id"):
        if len(group) <= seq_len:
            continue
        feature_matrix = group[feature_columns].values.astype("float32")
        target = group["Class"].values.astype("float32")
        for i in range(seq_len, len(feature_matrix)):
            sequences.append(feature_matrix[i - seq_len : i])
            labels.append(target[i])
    if not sequences:
        return None, None
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32).unsqueeze(1)


def train_autoencoder(
    features: torch.Tensor,
    hidden_dims=(64, 32),
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TransactionAutoencoder:
    device = torch.device(device)
    model = TransactionAutoencoder(features.size(1), hidden_dims=hidden_dims).to(device)
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
            total_count += batch.size(0)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            avg_loss = total_loss / max(total_count, 1)
            print(f"Autoencoder epoch {epoch}/{epochs} loss={avg_loss:.6f}")

    return model


def train_lstm(
    sequences: torch.Tensor,
    labels: torch.Tensor,
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TransactionLSTM:
    device = torch.device(device)
    model = TransactionLSTM(sequences.size(-1), hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    dataset = TensorDataset(sequences, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            total_count += x_batch.size(0)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            avg_loss = total_loss / max(total_count, 1)
            print(f"LSTM epoch {epoch}/{epochs} loss={avg_loss:.6f}")

    return model
