# Verifex
Fraud scoring and explainable transaction verification.

---

## What It Does
Verifex evaluates transactions with a backend fraud scoring API and a React frontend. It returns a risk score, verdict, and explanation for each transaction.

---

## Project Structure
```
verifex-verifex-clean/
├── backend/
│   ├── app.py             ← FastAPI API server
│   ├── train.py           ← Train models and build artifacts
│   ├── explain.py         ← Explanation logic
│   ├── gnn.py             ← Graph neural network model code
│   └── models.py          ← Autoencoder and LSTM model code
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/               ← React app source
├── requirements.txt       ← Python backend dependencies
└── README.md
```

---

## Quick Start

### 1. Create a Python virtual environment
This project requires Python 3.11.
Run the setup script below to create `.venv` and install backend dependencies.

```powershell
cd "c:\Users\Mahesh\Downloads\verifex-verifex-clean\verifex-verifex-clean"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

If Python 3.11 is not installed, the script attempts to install it using `winget`.

### 2. Prepare model artifacts
The backend now trains and saves PyTorch models as part of its artifact generation.
If you have `creditcard.csv` in the repo root, run:
```powershell
.\.venv\Scripts\python.exe backend/train.py
```
If you do not have the dataset, you can still run the backend once the required artifacts are available.

### 4. Start the backend API
```powershell
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### 5. Start the frontend
```powershell
cd frontend
npm install
npm run dev
```

### 6. Open the app
Visit the Vite URL shown in the frontend terminal, usually:

- `http://localhost:5173`

---

## Notes
- The frontend sends transaction data to `POST /score-transaction`.
- The backend also exposes `GET /health` for a quick health check.
- If `models_loaded` is `False`, training artifacts are missing or failed to load.

---

## Requirements
- Python 3.11+ recommended
- Node.js 18+ for frontend
- `creditcard.csv` is used by `backend/train.py` to generate training artifacts
- Windows users may need the Microsoft Visual C++ Redistributable installed for PyTorch
