import numpy as np
import torch
import shap
from typing import Dict, List

FEATURE_NAMES = ["Amount", "hour_of_day", "is_night"]

FEATURE_LABELS = {
    "Amount":      ("High transaction amount raises fraud risk",  "Low transaction amount reduces fraud risk"),
    "hour_of_day": ("Unusual transaction hour raises fraud risk", "Normal transaction hour reduces fraud risk"),
    "is_night":    ("Late-night timing flagged by ML model",      "Daytime timing reduces fraud risk"),
}


def _ae_predict(model):
    def predict(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t     = torch.tensor(X, dtype=torch.float32)
            recon = model(t)
            error = torch.mean(torch.abs(t - recon), dim=1).numpy()
        return np.clip(1.0 / (1.0 + np.exp(-np.clip((error - 0.04) * 20.0, -10.0, 10.0))), 0.02, 0.98)
    return predict


def _lstm_predict(model):
    def predict(X: np.ndarray) -> np.ndarray:
        seq = np.tile(X[:, np.newaxis, :], (1, 8, 1)).astype("float32")
        with torch.no_grad():
            out = model(torch.tensor(seq)).squeeze(-1).numpy()
        return np.clip(out, 0.02, 0.98)
    return predict


def build_background(scaler, n: int = 100) -> np.ndarray:
    rng     = np.random.default_rng(42)
    amounts = rng.exponential(scale=500, size=n).reshape(-1, 1)
    hours   = rng.integers(0, 24, size=n).reshape(-1, 1).astype(float)
    nights  = ((hours < 6) | (hours >= 23)).astype(float)
    raw     = np.hstack([amounts, hours, nights]).astype("float32")
    return scaler.transform(raw)


def _kernel_shap(predict_fn, background: np.ndarray, x: np.ndarray, nsamples: int = 64) -> np.ndarray:
    explainer = shap.KernelExplainer(predict_fn, background)
    sv        = explainer.shap_values(x.reshape(1, -1), nsamples=nsamples, silent=True)
    return np.array(sv).flatten()


def compute_shap(
    ae_model,
    lstm_model,
    scaler,
    feature_vec: np.ndarray,
    background: np.ndarray,
    ae_weight: float = 0.5,
    lstm_weight: float = 0.5,
) -> Dict:
    ae_sv   = _kernel_shap(_ae_predict(ae_model),     background, feature_vec) if ae_model   else np.zeros(3)
    lstm_sv = _kernel_shap(_lstm_predict(lstm_model),  background, feature_vec) if lstm_model else np.zeros(3)

    blended = ae_sv * ae_weight + lstm_sv * lstm_weight

    aggregated = {n: float(round(v, 5)) for n, v in zip(FEATURE_NAMES, blended)}
    dominant   = max(aggregated, key=lambda k: abs(aggregated[k]))

    ae_base   = float(_ae_predict(ae_model)(background).mean())    if ae_model   else 0.2
    lstm_base = float(_lstm_predict(lstm_model)(background).mean()) if lstm_model else 0.2
    base_value = round(ae_base * ae_weight + lstm_base * lstm_weight, 4)

    return {
        "aggregated":       aggregated,
        "per_model": {
            "autoencoder": {n: float(round(v, 5)) for n, v in zip(FEATURE_NAMES, ae_sv)},
            "lstm":        {n: float(round(v, 5)) for n, v in zip(FEATURE_NAMES, lstm_sv)},
        },
        "dominant_feature": dominant,
        "base_value":       base_value,
    }


def shap_reasons(aggregated: Dict[str, float], threshold: float = 0.03) -> List[str]:
    reasons = []
    for name, val in sorted(aggregated.items(), key=lambda x: -abs(x[1])):
        if abs(val) < threshold:
            continue
        pos, neg = FEATURE_LABELS.get(name, (f"{name} raises risk", f"{name} lowers risk"))
        reasons.append(pos if val > 0 else neg)
    return reasons