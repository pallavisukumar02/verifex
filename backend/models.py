import torch
import torch.nn as nn


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
