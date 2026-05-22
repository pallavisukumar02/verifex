from typing import Dict, List


def explain_transaction(
    transaction: Dict,
    profile: Dict,
    autoencoder_score: float,
    lstm_score: float,
    gnn_score: float,
) -> Dict:
    reasons: List[str] = []
    contributions = {}

    amount = transaction.get("Amount", 0.0)
    avg_amount = profile.get("avg_amount", 0.0)
    location = transaction.get("location")
    hour = transaction.get("hour_of_day", 0)
    device_id = transaction.get("device_id")
    known_devices = profile.get("known_devices", [])
    known_locations = profile.get("locations", [])

    if device_id and device_id not in known_devices:
        reasons.append("New device used for this account")
        contributions["new_device"] = 0.16
    if hour < 6 or hour >= 23:
        reasons.append("Late-night transaction")
        contributions["unusual_time"] = 0.12
    if avg_amount > 0 and amount > avg_amount * 5:
        reasons.append("Amount much higher than normal user spending")
        contributions["high_amount"] = 0.2
    elif amount >= 100000:
        reasons.append("Transaction amount is unusually large")
        contributions["large_amount"] = 0.25
    if location and location not in known_locations:
        reasons.append("Location differs from usual user geography")
        contributions["new_location"] = 0.14

    if autoencoder_score > 0.55:
        reasons.append("Autoencoder flags this transaction as anomalous")
        contributions["anomaly_detection"] = round((autoencoder_score - 0.5) * 0.4, 3)
    if lstm_score > 0.55:
        reasons.append("History model detects abnormal user behavior")
        contributions["history_pattern"] = round((lstm_score - 0.5) * 0.4, 3)
    if gnn_score > 0.55:
        reasons.append("Graph model detects potential network fraud")
        contributions["network_risk"] = round((gnn_score - 0.5) * 0.45, 3)

    if not reasons:
        reasons.append("Transaction is consistent with expected account behavior")
        contributions["baseline"] = -0.05

    model_score = (autoencoder_score + lstm_score + gnn_score) / 3.0
    rule_score = sum(value for value in contributions.values() if value > 0)
    blended = model_score * 0.5 + min(rule_score, 0.6) * 0.5
    risk_score = min(1.0, max(0.0, blended))

    if risk_score >= 0.65:
        verdict = "FRAUD"
    elif risk_score >= 0.45:
        verdict = "REVIEW"
    else:
        verdict = "LEGIT"

    return {
        "risk_score": round(risk_score, 4),
        "verdict": verdict,
        "reasons": reasons,
        "contributions": contributions,
    }
