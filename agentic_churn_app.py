"""
Agentic Churn Prediction App (LLM mapping) - improved and fixed.

- Place churn_pipe_2024.pkl in same folder.
- Set OPENAI_API_KEY in environment to enable LLM mapping (optional).
- Configure LLM_TIMEOUT_SECONDS env var (seconds) to control how long we wait for the LLM mapping (default 10s).
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, ORJSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import io
import pickle
import os
from dotenv import load_dotenv
import difflib
import traceback
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Optional OpenAI usage
try:
    import openai
except Exception:
    openai = None

# Load .env first so os.getenv picks values from the file
try:
    load_dotenv()
except Exception:
    pass

# ----------------- CONFIG (env-driven) -----------------
MODEL_PATH = os.getenv("MODEL_PATH", "./churn_pipe_2024.pkl")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # leave empty to skip LLM
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# LLM mapping timeout (seconds) and enable switch
try:
    LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "5"))
except Exception:
    LLM_TIMEOUT_SECONDS = 5.0
LLM_MAPPING_ENABLED = os.getenv("LLM_MAPPING_ENABLED", "false").lower() in ("1","true","yes")

# Prediction batching and parallelism
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

BATCH_SIZE = _int_env("PREDICT_BATCH_SIZE", 5000)
MAX_PARALLEL_WORKERS = _int_env("PREDICT_MAX_WORKERS", 4)
# ------------------------------------------------------------

# configure openai client only if enabled and key present
if LLM_MAPPING_ENABLED and OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY
else:
    openai = None

app = FastAPI(title="Agentic Churn Predictor (LLM mapping)")
try:
    from fastapi.middleware.gzip import GZipMiddleware
    app.add_middleware(GZipMiddleware, minimum_size=1024)
except Exception:
    pass

# Logging
logger = logging.getLogger("agentic_churn_app")
logging.basicConfig(level=logging.INFO)

# Globals to hold model loaded at startup
GLOBAL_MODEL = None
MODEL_LOAD_ERROR: Optional[str] = None
GLOBAL_MODEL_FEATURES: Optional[List[str]] = None
MODEL_LOCK = threading.Lock()
PREDICT_POOL: Optional[ThreadPoolExecutor] = None
MAPPING_CACHE: Dict[str, Dict[str, Optional[str]]] = {}


# ---------- Utilities ----------

def get_model_feature_names(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass
    try:
        from sklearn.compose import ColumnTransformer
    except Exception:
        ColumnTransformer = None
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            if ColumnTransformer and isinstance(step, ColumnTransformer):
                try:
                    cols = []
                    for t in step.transformers_:
                        if isinstance(t[2], (list, tuple, np.ndarray)):
                            cols.extend(list(t[2]))
                    if cols:
                        return cols
                except Exception:
                    pass
            if hasattr(step, "feature_names_in_"):
                try:
                    return list(step.feature_names_in_)
                except Exception:
                    pass
    for attr in ("feature_names", "features", "input_features"):
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                return list(val)
            except Exception:
                pass
    return None

def is_numberish(x: Any) -> bool:
    try:
        if x is None:
            return False
        s = str(x).strip()
        if s == '':
            return False
        float(s)
        return True
    except Exception:
        return False

def normalize_colname(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(' ', '_').replace('-', '_')
    keep = [c for c in s if (c.isalnum() or c == '_')]
    return ''.join(keep)

def fuzzy_match(target: str, candidates: List[str], cutoff: float = 0.6) -> Optional[str]:
    if not candidates:
        return None
    if target in candidates:
        return target
    matches = difflib.get_close_matches(target, candidates, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    tokens = set([t for t in target.split('_') if t])
    best = None
    best_score = 0.0
    for c in candidates:
        score = len(tokens.intersection(set(c.split('_')))) / max(1, len(tokens))
        if score > best_score:
            best_score = score
            best = c
    if best_score >= 0.5:
        return best
    return None

def map_model_features_to_df_columns(model_features: List[str], df_columns: List[str]) -> Dict[str, Optional[str]]:
    mapping = {}
    norm_candidates = [normalize_colname(c) for c in df_columns]
    for mf in model_features:
        mf_norm = normalize_colname(mf)
        # exact normalized match
        if mf_norm in norm_candidates:
            mapping[mf] = df_columns[norm_candidates.index(mf_norm)]
            continue
        # fuzzy on normalized names
        m = fuzzy_match(mf_norm, norm_candidates, cutoff=0.6)
        if m:
            mapping[mf] = df_columns[norm_candidates.index(m)]
            continue
        # fallback attempts
        attempts = [mf_norm.replace('_id', ''), mf_norm.replace('is_', ''), mf_norm.replace('has_', '')]
        found = None
        for a in attempts:
            m2 = fuzzy_match(a, norm_candidates, cutoff=0.55)
            if m2:
                found = m2
                break
        if found:
            mapping[mf] = df_columns[norm_candidates.index(found)]
            continue
        mapping[mf] = None
    return mapping

def coerce_and_impute(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dropna().apply(lambda x: is_numberish(x)).all():
        return pd.to_numeric(s, errors='coerce')
    if s.isnull().all():
        return s.fillna('missing')
    try:
        return s.fillna(s.mode()[0])
    except Exception:
        return s.fillna('missing')

def prepare_dataframe_for_model_using_mapping(df: pd.DataFrame, model_features: List[str], mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    aligned = pd.DataFrame()
    df_cols = list(df.columns)
    for feat in model_features:
        src = mapping.get(feat)
        if src is not None:
            if src not in df_cols:
                cand = next((c for c in df_cols if normalize_colname(c) == normalize_colname(src)), None)
                if cand:
                    src = cand
                else:
                    src = None
        if src is not None:
            series = df[src]
            series = coerce_and_impute(series)
            aligned[feat] = series
        else:
            low = feat.lower()
            created_series = None
            if 'age' in low:
                for cand in ['dob', 'birthdate', 'birth_date', 'date_of_birth', 'birth']:
                    if cand in df_cols:
                        try:
                            created_series = (pd.to_datetime('today') - pd.to_datetime(df[cand])).dt.days // 365
                            break
                        except Exception:
                            pass
            if created_series is None and (low.startswith('is_') or low.startswith('has_')):
                root = low.split('_', 1)[-1]
                candidate = fuzzy_match(root, [c.lower() for c in df_cols], cutoff=0.5)
                if candidate:
                    created_series = df[candidate].apply(lambda v: 1 if str(v).lower() in ('yes', 'true', '1') else 0)
            if created_series is None:
                if 'id' in low or feat.endswith('_id'):
                    created_series = pd.Series(range(len(df)))
                else:
                    created_series = pd.Series([0] * len(df))
            aligned[feat] = created_series
    aligned = aligned[model_features]
    return aligned

def run_model_prediction(model, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if hasattr(model, 'predict'):
        pred = model.predict(X)
    else:
        raise Exception('Model has no predict method')
    proba = None
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None
    return pred, proba

def _predict_batch(model, X_batch: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return run_model_prediction(model, X_batch)

def predict_in_batches_parallel(model, X: pd.DataFrame, batch_size: int, max_workers: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(X) == 0:
        return np.array([]), None
    batches = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batches.append((start, end))

    proba_detected = None
    preds_agg = [None] * len(batches)
    probas_agg = [None] * len(batches)

    # Use a pool (global) to avoid recreating threads each request
    global PREDICT_POOL
    if PREDICT_POOL is None or getattr(PREDICT_POOL, "_max_workers", None) != max_workers:
        try:
            PREDICT_POOL = ThreadPoolExecutor(max_workers=max_workers)
        except Exception:
            PREDICT_POOL = ThreadPoolExecutor(max_workers=1)

    futures = {}
    for i, (s, e) in enumerate(batches):
        Xb = X.iloc[s:e]
        fut = PREDICT_POOL.submit(_predict_batch, model, Xb)
        futures[fut] = i

    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            p, pr = fut.result()
            preds_agg[idx] = np.asarray(p)
            if pr is not None:
                pr_arr = np.asarray(pr)
                probas_agg[idx] = pr_arr
                proba_detected = True
            else:
                probas_agg[idx] = None
        except Exception:
            preds_agg[idx] = np.asarray([])
            probas_agg[idx] = None

    preds = np.concatenate([p for p in preds_agg if p is not None]) if any(p is not None for p in preds_agg) else np.array([])
    if proba_detected:
        probas_list = [pr for pr in probas_agg if pr is not None]
        if probas_list:
            try:
                probas = np.concatenate(probas_list, axis=0)
            except Exception:
                probas = None
        else:
            probas = None
    else:
        probas = None
    return preds, probas

# ----- Read uploaded bytes and detect header, read requested sheet if Excel -----
def detect_header_and_read(df_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    ext = filename.split('.')[-1].lower()
    try:
        if ext in ("xls", "xlsx"):
            xls = pd.ExcelFile(io.BytesIO(df_bytes))
            if sheet_name is not None and sheet_name != "":
                try:
                    sheet_idx = int(sheet_name)
                    sheet = xls.sheet_names[sheet_idx]
                except Exception:
                    if sheet_name in xls.sheet_names:
                        sheet = sheet_name
                    else:
                        sheet = xls.sheet_names[0]
            else:
                sheet = xls.sheet_names[0]
            raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        else:
            raw = pd.read_csv(io.BytesIO(df_bytes), header=None)
    except Exception:
        try:
            raw = pd.read_csv(io.BytesIO(df_bytes), header=None, encoding='latin1')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    if raw.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    candidate_headers = []
    max_rows = min(5, raw.shape[0])
    for r in range(max_rows):
        row = raw.iloc[r].astype(str).replace('nan', np.nan)
        non_nulls = row.dropna()
        non_numeric = sum(1 for v in non_nulls if not is_numberish(v))
        unique_vals = len(set(non_nulls))
        candidate_headers.append((r, non_numeric, unique_vals))
    candidate_headers.sort(key=lambda t: (t[1], t[2]), reverse=True)
    best_row = candidate_headers[0][0]
    header = raw.iloc[best_row].astype(str).fillna('').tolist()
    data = raw.iloc[best_row + 1 :].reset_index(drop=True)
    data.columns = [str(h) if str(h).strip() else f"col_{i}" for i, h in enumerate(header)]
    return data

# ----- LLM mapping helper (unchanged) -----
def ask_llm_for_mapping(model_features: List[str], sheet_columns: List[str], sample_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Optional[str]]]:
    if not openai:
        return None

    sample_preview = json.dumps(sample_rows, default=str, ensure_ascii=False, indent=2) if sample_rows else "[]"

    system = {
        "role": "system",
        "content": (
            "You are a precise data engineering assistant. You will be GIVEN two lists: REQUIRED model feature names and the COLUMN NAMES from an Excel sheet and a small SAMPLE of rows. "
            "Your job: identify which sheet columns correspond to each model feature and output a single JSON object (and nothing else). The JSON must have the keys: "
            "\"mapping\" (object) and optionally \"notes\" (string). \"mapping\" maps each model_feature -> sheet_column_name OR null if column not present. "
            "If the LLM thinks a model feature can be synthesized from existing columns, put the source column name and a short synthesis hint as the value object, like {\"source\":\"dob\",\"synth\":\"compute_age_from_dob\"}. Output STRICT JSON only — do not include any explanatory text."
        )
    }

    user_content = {
        "role": "user",
        "content": (
            f"Model features (list): {json.dumps(model_features, ensure_ascii=False)}\n\n"
            f"Sheet column names (list): {json.dumps(sheet_columns, ensure_ascii=False)}\n\n"
            f"Sample rows (up to 5, as list of objects):\n{sample_preview}\n\n"
            "Return JSON with shape: {\"mapping\": {\"model_feature\": <sheet_column_name | null | {\"source\":...,\"synth\":...}>}, \"notes\": \"optional notes\"}\n"
            "Remember: OUTPUT JSON ONLY."
        )
    }

    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[system, user_content],
            temperature=0.0,
            max_tokens=1000
        )
        text = resp["choices"][0]["message"]["content"].strip()
        try:
            parsed = json.loads(text)
        except Exception:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end+1])
                except Exception:
                    return None
            else:
                return None
        if "mapping" in parsed and isinstance(parsed["mapping"], dict):
            final_map = {}
            for mf, val in parsed["mapping"].items():
                if isinstance(val, str):
                    final_map[mf] = val
                elif val is None:
                    final_map[mf] = None
                elif isinstance(val, dict):
                    final_map[mf] = val
                else:
                    final_map[mf] = None
            return final_map
        else:
            return None
    except Exception:
        return None

# ----- Model loader used at startup (robust) -----
def _try_load_model(path: str):
    attempts = []
    missing_modules = set()

    def _record(name, exc):
        attempts.append((name, repr(exc)))
        if isinstance(exc, ModuleNotFoundError):
            if getattr(exc, "name", None):
                missing_modules.add(exc.name)
            else:
                s = str(exc)
                if "No module named" in s:
                    mod = s.split("No module named")[-1].strip().strip(" '\"")
                    if mod:
                        missing_modules.add(mod)

    # try pickle
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        _record("pickle.load", e)

    # try pickle with latin1 (py2->py3)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        _record("pickle.load(encoding='latin1')", e)

    # try joblib
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        _record("joblib.load", e)

    # try cloudpickle
    try:
        import cloudpickle
        with open(path, 'rb') as f:
            return cloudpickle.load(f)
    except Exception as e:
        _record("cloudpickle.load", e)

    # try dill
    try:
        import dill
        with open(path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        _record("dill.load", e)

    if missing_modules:
        mods = ", ".join(sorted(missing_modules))
        raise RuntimeError(f"Missing packages required to load model: {mods}")

    diag_lines = ["Model loading attempts failed; diagnostics:"]
    for name, err in attempts:
        diag_lines.append(f"- {name}: {err}")
    raise RuntimeError("\n".join(diag_lines))


@app.on_event("startup")
def load_model_on_startup():
    global GLOBAL_MODEL, MODEL_LOAD_ERROR, GLOBAL_MODEL_FEATURES
    with MODEL_LOCK:
        if GLOBAL_MODEL is not None or MODEL_LOAD_ERROR is not None:
            return
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"No model found at {MODEL_PATH}. Place churn_pipe_2024.pkl there.")
            logger.info("Loading model at startup from %s ...", MODEL_PATH)
            GLOBAL_MODEL = _try_load_model(MODEL_PATH)
            # Try to enable internal parallelism if estimator supports it
            try:
                if hasattr(GLOBAL_MODEL, 'n_jobs') and getattr(GLOBAL_MODEL, 'n_jobs') in (None, 1):
                    setattr(GLOBAL_MODEL, 'n_jobs', max(1, min(MAX_PARALLEL_WORKERS, os.cpu_count() or 1)))
            except Exception:
                pass
            GLOBAL_MODEL_FEATURES = get_model_feature_names(GLOBAL_MODEL)
            logger.info("Model loaded successfully.")
        except Exception as e:
            MODEL_LOAD_ERROR = str(e)
            traceback.print_exc()
            logger.error("Model load failed at startup: %s", MODEL_LOAD_ERROR)


# ----- Endpoint -----
@app.post('/predict-file')
async def predict_file(file: UploadFile = File(...), sheet_name: str = Form(None)):
    # 1) ensure model is ready
    if MODEL_LOAD_ERROR is not None:
        raise HTTPException(status_code=500, detail=f"Model failed to load at startup: {MODEL_LOAD_ERROR}")
    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet. Restart the app or check logs.")

    # 2) read file & detect header
    try:
        content = await file.read()
        df = detect_header_and_read(content, file.filename, sheet_name)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to read input file: {e}")

    # 3) model features
    model_features = GLOBAL_MODEL_FEATURES or get_model_feature_names(GLOBAL_MODEL)
    if model_features is None:
        model_features = list(df.columns)

    # 4) sample rows (avoid silent downcast warning)
    sample_rows = df.head(5).fillna('').astype(object).to_dict(orient='records')

    # 5) LLM mapping with timeout (fallback to heuristic)
    mapping = None
    used_mapping_source = "heuristic"

    if openai:
        result_container = {"mapping": None}

        def _call_llm():
            try:
                res = ask_llm_for_mapping(model_features, list(df.columns), sample_rows)
                result_container["mapping"] = res
            except Exception:
                result_container["mapping"] = None

        th = threading.Thread(target=_call_llm, daemon=True)
        th.start()
        th.join(timeout=LLM_TIMEOUT_SECONDS)
        if th.is_alive():
            logger.warning("LLM mapping timed out after %.1f seconds; falling back to heuristic mapping.", LLM_TIMEOUT_SECONDS)
            mapping = None
            used_mapping_source = "heuristic"
        else:
            mapping = result_container.get("mapping")
            if mapping:
                used_mapping_source = "llm"
            else:
                used_mapping_source = "heuristic"
    else:
        mapping = None
        used_mapping_source = "heuristic"

    sheet_cols = list(df.columns)
    # cache key based on model features + sheet columns signature
    cache_key = json.dumps({"mf": model_features, "sc": sheet_cols}, ensure_ascii=False)
    if mapping is None:
        cached = MAPPING_CACHE.get(cache_key)
        if cached:
            mapping = cached
            used_mapping_source = used_mapping_source + "+cache"
        else:
            mapping = map_model_features_to_df_columns(model_features, sheet_cols)
            # store in cache
            MAPPING_CACHE[cache_key] = mapping

    # 6) prepare aligned dataframe
    try:
        X_aligned = prepare_dataframe_for_model_using_mapping(df, model_features, mapping)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to prepare DataFrame for model: {e}")

    # 7) run prediction (batched + parallel)
    try:
        if len(X_aligned) > BATCH_SIZE:
            preds, probas = predict_in_batches_parallel(GLOBAL_MODEL, X_aligned, BATCH_SIZE, MAX_PARALLEL_WORKERS)
        else:
            preds, probas = run_model_prediction(GLOBAL_MODEL, X_aligned)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # 8) assemble output
    out = df.copy()
    try:
        out['prediction'] = list(preds)
    except Exception:
        out['prediction'] = preds

    if probas is not None:
        try:
            proba_arr = np.asarray(probas)
            if proba_arr.ndim == 2 and proba_arr.shape[1] == 2:
                out['probability'] = list(proba_arr[:, 1])
            else:
                for i in range(proba_arr.shape[1]):
                    out[f'prob_class_{i}'] = list(proba_arr[:, i])
        except Exception:
            traceback.print_exc()

    serial_mapping = {}
    for k, v in mapping.items():
        if isinstance(v, (str, type(None))):
            serial_mapping[k] = v
        elif isinstance(v, dict):
            serial_mapping[k] = v
        else:
            serial_mapping[k] = None

    # sanitize and convert to JSON-friendly structures
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notnull(out), None)
    predictions_jsonable = jsonable_encoder(out.to_dict(orient='records'))

    return ORJSONResponse({
        'n_rows': len(out),
        'mapping': serial_mapping,
        'mapping_source': used_mapping_source,
        'predictions': predictions_jsonable
    })


def _dataframe_to_csv_stream(df: pd.DataFrame):
    buf = io.StringIO()
    # write header
    df.head(0).to_csv(buf, index=False)
    yield buf.getvalue()
    buf.seek(0)
    buf.truncate(0)
    # stream body in chunks
    chunk_size = 100000
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        df.iloc[start:end].to_csv(buf, index=False, header=False)
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)


@app.post('/predict-file-csv')
async def predict_file_csv(file: UploadFile = File(...), sheet_name: str = Form(None)):
    # Reuse the same logic but stream CSV instead of JSON
    if MODEL_LOAD_ERROR is not None:
        raise HTTPException(status_code=500, detail=f"Model failed to load at startup: {MODEL_LOAD_ERROR}")
    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet. Restart the app or check logs.")

    try:
        content = await file.read()
        df = detect_header_and_read(content, file.filename, sheet_name)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to read input file: {e}")

    model_features = GLOBAL_MODEL_FEATURES or get_model_feature_names(GLOBAL_MODEL)
    if model_features is None:
        model_features = list(df.columns)

    sample_rows = df.head(5).fillna('').astype(object).to_dict(orient='records')

    mapping = None
    if openai:
        result_container = {"mapping": None}
        def _call_llm():
            try:
                res = ask_llm_for_mapping(model_features, list(df.columns), sample_rows)
                result_container["mapping"] = res
            except Exception:
                result_container["mapping"] = None
        th = threading.Thread(target=_call_llm, daemon=True)
        th.start()
        th.join(timeout=LLM_TIMEOUT_SECONDS)
        if not th.is_alive():
            mapping = result_container.get("mapping")

    sheet_cols = list(df.columns)
    cache_key = json.dumps({"mf": model_features, "sc": sheet_cols}, ensure_ascii=False)
    if mapping is None:
        cached = MAPPING_CACHE.get(cache_key)
        if cached:
            mapping = cached
        else:
            mapping = map_model_features_to_df_columns(model_features, sheet_cols)
            MAPPING_CACHE[cache_key] = mapping

    try:
        X_aligned = prepare_dataframe_for_model_using_mapping(df, model_features, mapping)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to prepare DataFrame for model: {e}")

    try:
        if len(X_aligned) > BATCH_SIZE:
            preds, probas = predict_in_batches_parallel(GLOBAL_MODEL, X_aligned, BATCH_SIZE, MAX_PARALLEL_WORKERS)
        else:
            preds, probas = run_model_prediction(GLOBAL_MODEL, X_aligned)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    out = df.copy()
    try:
        out['prediction'] = list(preds)
    except Exception:
        out['prediction'] = preds
    if probas is not None:
        try:
            proba_arr = np.asarray(probas)
            if proba_arr.ndim == 2 and proba_arr.shape[1] == 2:
                out['probability'] = list(proba_arr[:, 1])
            else:
                for i in range(proba_arr.shape[1]):
                    out[f'prob_class_{i}'] = list(proba_arr[:, i])
        except Exception:
            traceback.print_exc()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notnull(out), None)

    headers = {"Content-Disposition": f"attachment; filename=predictions.csv"}
    return StreamingResponse(_dataframe_to_csv_stream(out), media_type='text/csv', headers=headers)


@app.get('/')
async def root():
    return {
        'service': 'Agentic Churn Predictor (LLM mapping)',
        'notes': 'POST /predict-file (file form-data + sheet_name Form field). Model loaded at startup from churn_pipe_2024.pkl in same folder.'
    }