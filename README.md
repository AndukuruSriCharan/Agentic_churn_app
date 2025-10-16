# Agentic Churn Predictor (LLM Mapping) — Fast, Batched, and Parallel

## Overview
- Backend: FastAPI service in `agentic_churn_app.py`
- Frontend: Streamlit UI in `streamlit_frontend.py`
- Model: Scikit-learn pipeline pickle at `churn_pipe_2024.pkl`
- Optimizations: batching, parallel inference, mapping cache, gzip, fast JSON; CSV streaming endpoint for large datasets

## Requirements
- Python 3.10+
- A virtual environment (recommended)
- The model file `churn_pipe_2024.pkl` in the project root

## Environment (.env)
Create a `.env` file in the project root:
```ini
MODEL_PATH=./churn_pipe_2024.pkl

# LLM (optional)
LLM_MAPPING_ENABLED=false
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=5

# Performance tuning
PREDICT_BATCH_SIZE=5000
PREDICT_MAX_WORKERS=4
```
Notes:
- Set `LLM_MAPPING_ENABLED=true` and paste your real `OPENAI_API_KEY` to enable LLM-based column mapping.
- For big files (e.g., 300k rows): increase `PREDICT_BATCH_SIZE` (e.g., 25000) and `PREDICT_MAX_WORKERS` to match CPU cores (e.g., 8).

## Install and Run
1. Activate venv (Windows):
```powershell
cd D:\P\Cur
.\venv\Scripts\activate
```
2. Install packages:
```powershell
pip install -U fastapi uvicorn[standard] python-dotenv pandas numpy scikit-learn openpyxl requests streamlit orjson
```
3. Start backend (FastAPI):
```powershell
uvicorn agentic_churn_app:app --host 0.0.0.0 --port 8000
```
- Health: open `http://localhost:8000/`

4. Start frontend (Streamlit) in another terminal:
```powershell
cd D:\P\Cur
.\venv\Scripts\activate
streamlit run streamlit_frontend.py
```
- UI: `http://localhost:8501`

## Endpoints
- `POST /predict-file` — returns JSON with mapping and predictions.
  - Use for small-to-medium results. Large responses (hundreds of thousands of rows) can be slow.
- `POST /predict-file-csv` — streams CSV back to the client.
  - Best choice for very large datasets; avoids huge JSON and reduces timeout risk.

Request form-data fields:
- `file`: the uploaded Excel/CSV file
- `sheet_name` (optional): sheet name or index (Excel); ignored for CSV

## Full Flow
1. Upload Excel/CSV in the UI.
2. UI optionally previews the first 10 rows of the selected sheet.
3. UI sends the file and `sheet_name` to the backend endpoint.
4. Backend detects header row, sanitizes columns, and loads the model.
5. Mapping step:
   - If LLM is enabled and available, backend asks the LLM for feature-to-column mapping with a small sample and a short timeout.
   - Otherwise, a heuristic mapping is used.
   - Results are cached per (model_features + sheet_columns) signature.
6. Backend builds an aligned DataFrame according to the model features and imputes/coerces values.
7. Prediction step:
   - Uses batching and a thread pool to parallelize inference for large datasets.
   - If the estimator supports `n_jobs`, it is set up to available CPUs (bounded by `PREDICT_MAX_WORKERS`).
8. Response:
   - JSON path: returns mapping plus predictions in JSON (with gzip and orjson optimization).
   - CSV path: streams a CSV file containing input columns + prediction columns.

## Using the Streamlit UI
- Set the backend URL in the sidebar (default `http://localhost:8000`).
- Adjust the “Request timeout (seconds)” for large jobs (e.g., 300–480s for 300k rows).
- Buttons:
  - “Run prediction (uses server-side model)” → calls JSON endpoint `/predict-file`.
  - “Run and download CSV from server” → calls CSV endpoint `/predict-file-csv` and gives you a CSV download.
- The UI shows mapping JSON (when using JSON endpoint), a preview of predictions, and download buttons.

## Performance Tuning
- `.env` suggestions for ~300k rows:
```ini
LLM_MAPPING_ENABLED=false
LLM_TIMEOUT_SECONDS=3
PREDICT_BATCH_SIZE=25000
PREDICT_MAX_WORKERS=8
```
- Keep LLM disabled for speed unless needed; if enabled, use `LLM_TIMEOUT_SECONDS` 3–5.
- If memory is limited, lower `PREDICT_BATCH_SIZE`.
- For maximum throughput, keep a single backend process; the internal thread pool handles parallelism.

## Troubleshooting
- ReadTimeout in UI:
  - Prefer the CSV endpoint/button for large results.
  - Increase the UI timeout slider (300–480s).
  - Ensure backend is reachable at `http://localhost:8000/`.
- Mapping looks wrong:
  - Enable LLM mapping and keep timeout small (3–5).
  - Test with a small file to inspect mapping JSON in UI.
- Slow predictions:
  - Increase `PREDICT_BATCH_SIZE` and `PREDICT_MAX_WORKERS` according to CPU/RAM.
  - Ensure LLM is disabled for production runs unless mapping is required.

## Security
- Do not hardcode secrets. Use the `.env` file and environment variables.
- The backend only loads OpenAI when `LLM_MAPPING_ENABLED=true` and a key is provided.

## Files
- `agentic_churn_app.py`: FastAPI backend (batching, parallelism, CSV streaming, gzip, orjson)
- `streamlit_frontend.py`: Streamlit UI (preview, JSON or CSV flow, timeout control)
- `churn_pipe_2024.pkl`: Model pickle (place in project root)

## License
Internal use or as permitted by your organization.
