# Quickstart (Local)

## 1) Create + activate `.venv`

Run from the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install pip packages

```bash
pip install -r requirements-prediction.txt -r requirements-insights.txt
# (or: pip install -r requirements.txt)
```

## 3) Run both servers (two terminals)

### Terminal A — AI/ML `predictionAPI` (port `5000`)

```bash
source .venv/bin/activate
cd scripts
python predictionAPI.py
```

### Terminal B — Insights API (port `5001`)

```bash
source .venv/bin/activate
cd scripts
flask --app insightsAPI:app run --host 0.0.0.0 --port 5001
```

