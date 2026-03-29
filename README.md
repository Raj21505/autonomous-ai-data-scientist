# AI Data Scientist - End-to-End Data Cleaning, EDA, Modeling, and Dashboard

An end-to-end web application that lets you:
- Upload a dataset (CSV/XLS/XLSX)
- Inspect data quality and feature types
- Run automated cleaning
- Generate EDA visualizations
- Train multiple ML models (classification or regression)
- Run single and batch predictions
- View an interactive dashboard generated from model and data insights

## What This Project Does

This project combines a FastAPI backend with a static HTML/CSS/JS frontend.

Pipeline flow:
1. Upload dataset
2. Automatic dataset understanding
3. Cleaning + feature filtering + encoding
4. EDA graph generation
5. Model training and best-model selection
6. Prediction (single row or batch file)
7. Interactive dashboard generation

## Tech Stack

- Backend: FastAPI, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- Frontend: Vanilla HTML/CSS/JavaScript
- Dashboard rendering: Plotly
- File support: CSV, XLS, XLSX

## Project Structure

```text
requirements.txt
backend/
  main.py
  state.py
  agents/
    data_cleaning.py
    data_understanding.py
    eda_agent.py
    feature_selection.py
    model_training.py
    dashboard_schema_generator.py
  utils/
    file_loader.py
    categorical_encoder.py
frontend/
  index.html
  cleaning.html
  target.html
  eda.html
  model.html
  dashboard.html
  css/style.css
  js/app.js
  js/eda.js
  js/model.js
  js/dashboard.js
```

## Prerequisites

- Python 3.9+ recommended
- pip
- A browser (Chrome/Edge recommended)

## Setup and Run

### 1) Clone and open project

Open the project root in VS Code (the folder that contains `requirements.txt`).

### 2) Create virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

### 4) Start backend API

From project root:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URL:
- http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

### 5) Start frontend static server

Open a second terminal and run:

```powershell
cd frontend
python -m http.server 5500
```

Frontend URL:
- http://127.0.0.1:5500/index.html

Important:
- Keep backend running on port 8000
- If frontend runs on a separate local static server (for example port 5500), frontend JS auto-calls `http://localhost:8000`
- If frontend is served by FastAPI (same origin), frontend JS auto-calls relative API routes

## Deploy On Render (Frontend + Backend Linked)

This repository is now configured so FastAPI serves both:
- API endpoints (for example `/upload`, `/clean`, `/train-models`)
- Frontend pages/assets from `frontend/` on the same Render URL

### 1) Push your latest code

Make sure these files are in your repo:
- `render.yaml`
- `backend/main.py` (now mounts `frontend/` as static files)
- Updated frontend JS files in `frontend/js/`

### 2) Create service from Blueprint

In Render:
1. Open Dashboard -> New -> Blueprint
2. Select your GitHub repo
3. Render reads `render.yaml` and creates the web service

`render.yaml` uses:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### 3) Open the deployed app

After deploy:
- App URL shows the frontend home page
- API docs are available at `/docs`

Because frontend and backend share one Render origin, there is no cross-origin URL mismatch in production.

### 4) Optional environment override

Frontend also supports an optional runtime global variable `window.API_BASE_URL`.
If you later split frontend/backend into separate services, you can inject that value to point frontend to a different backend URL.

## How to Use the App

1. Open `index.html`
2. Upload a dataset (CSV/XLS/XLSX)
3. On overview page, inspect rows/columns/missing values and select target column
4. Click analyze/clean to generate cleaning report
5. Open EDA page to generate and view graph sections
6. Open Model page:
   - Choose target
   - Train models
   - Review best model and comparison metrics
7. Run predictions:
   - Single prediction by entering feature values
   - Batch prediction by uploading a CSV file
8. Open Dashboard page for interactive KPIs/charts

## Backend API Endpoints

### Dataset and Cleaning

- `POST /upload`
  - Upload dataset file
  - Returns dataset id and initial analysis

- `POST /clean`
  - Form fields: `id`, `target`
  - Runs cleaning pipeline and returns cleaning report

- `GET /data/{id}`
  - Returns sample rows
  - Optional query: `full=1` to download full cleaned CSV

- `GET /categorical-info/{id}`
  - Returns detected categorical mappings and removed categorical columns

### EDA

- `POST /eda/{id}`
  - Form field: `target` (optional)
  - Generates EDA summary and stores plot images in memory

- `GET /eda-image/{id}/{category}/{sub_key}`
  - Returns a specific EDA image (PNG)

### Model Training and Results

- `POST /train-models`
  - Form fields: `id`, `target`
  - Trains multiple models and selects best model

- `GET /model-results/{id}`
  - Returns best model metrics, comparison, feature importance

### Dashboard

- `POST /generate-dashboard`
  - Form field: `id`
  - Builds dashboard schema from trained model + cleaned data

- `GET /dashboard-schema/{id}`
  - Query: `refresh=1` to regenerate schema
  - Otherwise returns cached schema when available

### Prediction

- `POST /predict`
  - Form fields:
    - `id`
    - `input_data` (JSON string with feature values)

- `POST /predict-batch`
  - Form fields:
    - `id`
    - `file` (CSV)
  - Returns row-wise predictions

## Data and Model State

Current implementation keeps runtime state in memory:
- Uploaded/cleaned datasets in `backend/state.py`
- Trained models and results in `backend/state.py`

This means:
- Data is lost when backend restarts
- Not suitable for production scaling as-is

## Supported ML Behavior

- Auto-detects task type:
  - Classification for non-numeric/discrete targets
  - Regression for continuous numeric targets
- Trains multiple candidate models and selects best using default metric:
  - Classification: F1-score
  - Regression: R2-score

## Common Issues and Fixes

### 1) CORS or network errors in frontend

- Ensure backend is running on port 8000
- Ensure frontend is served via HTTP (not just opening file directly)

### 2) `ModuleNotFoundError` or dependency issues

- Activate venv
- Re-run:

```powershell
pip install -r requirements.txt
```

### 3) Excel upload errors

- Confirm `openpyxl` is installed
- Ensure file extension matches file format

### 4) Training fails for all models

Possible causes:
- Target column not present after cleaning
- Too few valid rows after cleaning
- Target has single class only (for classification)

Try:
- Choosing another target column
- Reviewing cleaning report for heavy row/column removal

### 5) Dashboard not loading

- Train models first
- Check `/model-results/{id}` works in browser or Swagger
- Use refresh on dashboard schema endpoint

## Development Notes

- Backend entrypoint: `backend/main.py`
- Frontend API base URL is environment-aware:
  - same-origin in production (Render)
  - `http://localhost:8000` fallback for separate local static frontend servers

## Suggested Next Improvements

- Persist datasets/models to disk or database
- Add authentication and user-specific sessions
- Add logging and structured error handling
- Add unit/integration tests for agents and endpoints
- Add Docker setup for one-command startup

## License

Add your preferred license here (MIT/Apache-2.0/etc.).
