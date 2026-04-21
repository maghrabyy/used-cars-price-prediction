# Project Walkthrough And Setup

## What is in this repository

This project has four main parts:

1. `dashboard/`
   The Dash web UI where a user selects car features and gets a predicted price.

2. `scripts/`
   The prediction backend:
   - `predictionAPI.py` exposes a Flask REST API at `/predict_car_price`
   - `pricePrediction.py` loads the trained model and preprocessing assets from `models/`

3. `models/`
   Saved training artifacts used at runtime:
   - `trained_model.h5`
   - `trained_scaler.pkl`
   - `ohe_features.pkl`
   - `ohe_fuel.pkl`

4. `dag/`
   Airflow DAGs and SQL scripts for scraping Hatla2ee data, uploading it to S3, and loading it into Amazon Redshift.

There are also:

- `notebooks/` for analysis and model-building work
- `images/` for charts/screenshots used in the README/demo

## How the app works end-to-end

1. The Dash app in `dashboard/app.py` loads feature metadata from `models/ohe_features.pkl` and `models/ohe_fuel.pkl`.
2. When the user clicks `Predict Price`, Dash sends a POST request to `http://127.0.0.1:5001/predict_car_price`.
3. The Flask API in `scripts/predictionAPI.py` validates the JSON request and calls `predict_car_price(...)`.
4. `scripts/pricePrediction.py` loads the saved Keras model and scaler, encodes the input, and returns the predicted price.
5. The dashboard displays the estimated price together with the average market price for the selected model.

## Recommended setup

Use Python 3.11 in a virtual environment. Python 3.13 is not a good fit for this repo because the TensorFlow/Keras stack used by the saved model is not pinned for it here. Run these commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the insights API

The API uses relative paths, so start it from the `scripts/` directory:

```bash
cd scripts
flask --app insightsAPI:app run --port 5001 --host 0.0.0.0
```

Expected URL:

```text
http://127.0.0.1:5001/predict_car_price
```

## Run the dashboard

Open a second terminal, activate the same virtual environment, and start Dash from the `dashboard/` directory:

```bash
source .venv/bin/activate
cd dashboard
python app.py
```

Dash usually starts on:

```text
http://127.0.0.1:8050
```

Important: the API must already be running on port `5001`, because the dashboard sends prediction requests to that address.

## Quick usage flow

1. Start the Flask API.
2. Start the Dash dashboard.
3. Open the Dash URL in your browser.
4. Choose `Brand`, `Model`, `Class`, `Model year`, `Transmission`, `Fuel`, and `km`.
5. Click `Predict Price`.

## Test the API directly with curl

Run this after starting the API:

```bash
curl -X POST http://127.0.0.1:5001/predict_car_price \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Hyundai",
    "model": "accent",
    "year": "2018",
    "km": 80000,
    "transmission": "Automatic",
    "fuel": "gas",
    "class": "Standard"
  }'
```

## Airflow pipeline setup

The Airflow part is optional if your goal is only to run predictions. It is needed only if you want to scrape data and load Redshift/S3.

Install Airflow dependencies:

```bash
pip install -r requirements-airflow.txt
```

What the DAGs do:

- `dag/full_load_dag.py`: one-time bootstrap load
- `dag/incremental_load_dag.py`: daily incremental refresh
- `dag/helpers.py`: scraping tasks plus S3 upload helper
- `dag/sql/init_db_schema.sql`: creates raw, staging, and production schemas/tables
- `dag/sql/full_load.sql`: initial population from raw to staging/prod
- `dag/sql/incremental_load.sql`: refresh/update logic for daily loads
- `dag/sql/truncate.sql`: clears raw tables before reload

Before these DAGs can work, you must configure in Airflow:

- an AWS connection with ID `aws-connection`
- an S3 hook/connection compatible with `S3Hook('s3-bucket')`
- a Redshift connection named `redshift-cluster` for full load
- a Redshift connection named `cars-redshift` for incremental load
- an accessible Redshift cluster/database/user matching the SQL operator settings
- an S3 bucket named `used-cars-egypt-data`, or update the DAG constants

Typical local Airflow bootstrap commands:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
airflow db migrate
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin
airflow webserver --port 8080
```

In another terminal:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
airflow scheduler
```

To expose the DAGs to Airflow, point `AIRFLOW__CORE__DAGS_FOLDER` to this repo's `dag/` directory or copy/symlink the DAG files there.

## Important run assumptions

- `scripts/predictionAPI.py` should be started from `scripts/` because `pricePrediction.py` uses `../models/...` relative paths.
- `dashboard/app.py` should be started from `dashboard/` because it reads `../models/...`.
- The dashboard hardcodes the API URL as `http://127.0.0.1:5001/predict_car_price`.
- The repo does not currently include a lockfile, Docker setup, or automated tests.

## Useful commands summary

Create env and install app dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run insights API:

```bash
cd scripts
flask --app insightsAPI:app run --port 5001
```

Run dashboard:

```bash
cd dashboard
python app.py
```

Install optional Airflow stack:

```bash
pip install -r requirements-airflow.txt
```
