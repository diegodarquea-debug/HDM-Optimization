"""
Data loading and preprocessing from CSV (and BigQuery in future).
Auto-detects partner_id and date ranges.
"""
import concurrent.futures
import logging
import time
import numpy as np
from pathlib import Path
from datetime import date, datetime
from typing import Optional, Tuple, List, Union, Any, Dict
import pandas as pd
from .config import (
    RAW_DATA_PATH,
    GCP_PROJECT_ID,
    BQ_DATASET,
    BQ_TABLE,
    BQ_LOCATION,
    BQ_TIMEOUT_SECONDS,
    BQ_DEFAULT_LOOKBACK_DAYS,
    DATA_SOURCE,
)

logger = logging.getLogger(__name__)


def _resolve_data_source(data_source: Optional[str]) -> str:
    source = (data_source or DATA_SOURCE or "csv").lower()
    if source not in {"auto", "csv", "bigquery"}:
        raise ValueError(f"Invalid data source: {source}. Use auto, csv, or bigquery.")

    if source == "auto":
        has_bq_config = bool(GCP_PROJECT_ID and BQ_DATASET and BQ_TABLE)
        return "bigquery" if has_bq_config else "csv"

    return source


def _infer_bigquery_parameter(name: str, value: Any, bigquery_module: Any):
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return bigquery_module.ScalarQueryParameter(name, "DATE", value.date())

    if isinstance(value, datetime):
        return bigquery_module.ScalarQueryParameter(name, "DATE", value.date())

    if isinstance(value, date):
        return bigquery_module.ScalarQueryParameter(name, "DATE", value)

    if isinstance(value, bool):
        return bigquery_module.ScalarQueryParameter(name, "BOOL", value)

    if isinstance(value, int):
        return bigquery_module.ScalarQueryParameter(name, "INT64", value)

    if isinstance(value, float):
        return bigquery_module.ScalarQueryParameter(name, "FLOAT64", value)

    if isinstance(value, (list, tuple)):
        values = [item for item in value if item is not None]
        if not values:
            return bigquery_module.ArrayQueryParameter(name, "STRING", [])
        if all(isinstance(item, int) for item in values):
            return bigquery_module.ArrayQueryParameter(name, "INT64", values)
        return bigquery_module.ArrayQueryParameter(name, "STRING", [str(item) for item in values])

    return bigquery_module.ScalarQueryParameter(name, "STRING", str(value))

def load_bigquery_data(query: Optional[str] = None,
                       start_date: Optional[pd.Timestamp] = None,
                       end_date: Optional[pd.Timestamp] = None,
                       partner_ids: Optional[List[Any]] = None,
                       query_parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load data from Google BigQuery using Default Application Credentials.
    
    Supports parameterized queries with @start_date, @end_date, @partner_ids.
    
    Parameters
    ----------
    query: optional custom SQL query (else uses default hardcoded query)
    start_date, end_date: date filters
    partner_ids: list of partner IDs to filter (array parameter)
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        logger.error("google-cloud-bigquery not installed.")
        raise ImportError("Please install google-cloud-bigquery to use BigQuery loading.")

    resolved_project = GCP_PROJECT_ID

    if query is None and (not resolved_project or not BQ_DATASET or not BQ_TABLE):
        raise ValueError(
            "BigQuery configuration missing for default table query. Set GCP_PROJECT_ID (or GOOGLE_CLOUD_PROJECT), "
            "BQ_DATASET and BQ_TABLE, or provide a custom SQL query with --bq-query-file."
        )

    if query is not None and not resolved_project:
        raise ValueError(
            "Missing GCP project ID for BigQuery client initialization. "
            "Set GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT in your environment."
        )

    if query is not None and (not BQ_DATASET or not BQ_TABLE):
        logger.info("Using custom SQL query without BQ_DATASET/BQ_TABLE defaults.")

    query_params = []
    
    if query is None:
        # Default query (no parameters) — lookback controlled by BQ_DEFAULT_LOOKBACK_DAYS
        query = f"""
            SELECT *
            FROM `{resolved_project}.{BQ_DATASET}.{BQ_TABLE}`
            WHERE momento_exacto >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {BQ_DEFAULT_LOOKBACK_DAYS} DAY)
        """
    else:
        # Custom query: prepare parameters if referenced
        if "@start_date" in query and start_date is not None:
            sd = pd.Timestamp(start_date).date()
            query_params.append(bigquery.ScalarQueryParameter("start_date", "DATE", sd))
        if "@end_date" in query and end_date is not None:
            ed = pd.Timestamp(end_date).date()
            query_params.append(bigquery.ScalarQueryParameter("end_date", "DATE", ed))
        if "@partner_ids" in query:
            # Always pass partner_ids, even if empty (required for SQL logic)
            pids = [int(pid) for pid in (partner_ids or []) if pid]
            query_params.append(bigquery.ArrayQueryParameter("partner_ids", "INT64", pids))

    for param_name, param_value in (query_parameters or {}).items():
        placeholder = f"@{param_name}"
        if placeholder not in query:
            continue
        if param_name in {"start_date", "end_date", "partner_ids"}:
            continue
        inferred = _infer_bigquery_parameter(param_name, param_value, bigquery)
        if inferred is not None:
            query_params.append(inferred)

    # Use Default Application Credentials (environment auto-detection)
    try:
        client = bigquery.Client(project=resolved_project)
    except OSError as exc:
        raise OSError(
            "BigQuery client could not determine a project. "
            "Set GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT before running the pipeline."
        ) from exc
    
    job_config = bigquery.QueryJobConfig(query_parameters=query_params) if query_params else None
    logger.info(f"Executing BigQuery query with {len(query_params)} parameters...")
    logger.info(
        "BigQuery execution settings: project=%s, location=%s, timeout=%ss",
        resolved_project,
        BQ_LOCATION,
        BQ_TIMEOUT_SECONDS,
    )
    
    # Log parameter details (INFO level so user sees them)
    for param in query_params:
        param_val = param.values if hasattr(param, 'values') else param.value
        logger.info(f"  → Parameter: {param.name} = {param_val}")

    query_job = client.query(query, job_config=job_config, location=BQ_LOCATION)
    job_id = getattr(query_job, "job_id", "unknown")
    logger.info("BigQuery job submitted: job_id=%s, location=%s", job_id, BQ_LOCATION)
    logger.info(
        "Track status in CLI: bq --location=%s show -j %s",
        BQ_LOCATION,
        job_id,
    )

    start_wait = time.monotonic()
    poll_interval_seconds = 30
    while True:
        elapsed = int(time.monotonic() - start_wait)
        remaining = BQ_TIMEOUT_SECONDS - elapsed
        if remaining <= 0:
            logger.error(
                "BigQuery query timed out after %s seconds (job_id=%s).",
                BQ_TIMEOUT_SECONDS,
                job_id,
            )
            raise TimeoutError(
                "BigQuery query exceeded timeout. "
                f"job_id={job_id}. "
                "Increase BQ_TIMEOUT_SECONDS (for example: export BQ_TIMEOUT_SECONDS=1800), "
                "test with a shorter date range, "
                "or check status with: "
                f"bq --location={BQ_LOCATION} show -j {job_id}."
            )

        try:
            # Poll with short blocking windows so we can emit periodic progress logs.
            query_job.result(timeout=min(poll_interval_seconds, remaining))
            break
        except concurrent.futures.TimeoutError:
            logger.info(
                "BigQuery query still running... job_id=%s, elapsed=%ss, remaining=%ss",
                job_id,
                elapsed,
                remaining,
            )

    df = query_job.to_dataframe(create_bqstorage_client=False)
    
    if "momento_exacto" in df.columns:
        df["momento_exacto"] = pd.to_datetime(df["momento_exacto"])

    logger.info(f"Loaded {len(df)} rows from BigQuery")
    return df


def load_csv_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load data from CSV file."""
    if filepath is None:
        filepath = RAW_DATA_PATH
    
    if not Path(filepath).exists():
        logger.error(f"Data file not found: {filepath}")
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df["momento_exacto"] = pd.to_datetime(df["momento_exacto"])
    
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def get_unique_partners(df: pd.DataFrame) -> np.ndarray:
    """Get unique partner IDs from dataframe."""
    return df["partner_id"].unique()


def get_date_range(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get min and max dates from dataframe."""
    return df["momento_exacto"].min(), df["momento_exacto"].max()


def filter_by_partner(df: pd.DataFrame, partner_id: Any) -> pd.DataFrame:
    """Filter dataframe by partner_id."""
    return df[df["partner_id"] == partner_id].copy()


def filter_by_date_range(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Filter dataframe by date range."""
    return df[(df["momento_exacto"] >= start_date) & 
              (df["momento_exacto"] <= end_date)].copy()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data: handle missing values, validate columns, etc."""
    df = df.copy()
    
    required_cols = ["momento_exacto", "partner_id", "ordenes_pendientes", 
                     "riders_cerca", "hdm_activo", "max_awt_espera_min"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Handle missing values
    df["ordenes_pendientes"] = df["ordenes_pendientes"].fillna(0)
    df["riders_cerca"] = df["riders_cerca"].fillna(0)
    df["hdm_activo"] = df["hdm_activo"].fillna(0).astype(int)
    df["max_awt_espera_min"] = df["max_awt_espera_min"].fillna(0)
    
    # Ensure numeric columns
    numeric_cols = ["ept_promedio", "ept_promedio_min", "ept_configurado_min", "awt_promedio"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # EPT logic
    if "ept_promedio_min" in df.columns:
        df.loc[df["ordenes_pendientes"] == 0, "ept_promedio_min"] = 0.0
        active_mask = df["ordenes_pendientes"] > 0
        if active_mask.any():
            ept_p99 = df.loc[active_mask, "ept_promedio_min"].quantile(0.99)
            df.loc[active_mask, "ept_promedio_min"] = df.loc[active_mask, "ept_promedio_min"].clip(lower=0, upper=ept_p99)

        df["ept_promedio_min_smoothed"] = df["ept_promedio_min"].rolling(window=5, min_periods=1).mean()
        df.loc[df["ordenes_pendientes"] == 0, "ept_promedio_min_smoothed"] = 0.0
        df["ept_promedio"] = df["ept_promedio_min_smoothed"]
    
    df = df.sort_values("momento_exacto").reset_index(drop=True)
    logger.info("Data preprocessing complete.")
    return df


def load_and_prepare_data(filepath: Optional[str] = None,
                          partner_id: Optional[Any] = None,
                          start_date: Optional[pd.Timestamp] = None,
                          end_date: Optional[pd.Timestamp] = None,
                          mode: str = "partner",
                          source: Optional[str] = None,
                          bigquery_query: Optional[str] = None,
                          partner_ids: Optional[List[Any]] = None,
                          query_parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Complete pipeline: load → preprocess → filter."""
    resolved_source = _resolve_data_source(source)

    if resolved_source == "bigquery":
        resolved_partner_ids = list(partner_ids or [])
        if mode == "partner" and partner_id is not None and partner_id not in resolved_partner_ids:
            resolved_partner_ids.append(partner_id)

        df = load_bigquery_data(
            query=bigquery_query,
            start_date=start_date,
            end_date=end_date,
            partner_ids=resolved_partner_ids,
            query_parameters=query_parameters,
        )
    else:
        df = load_csv_data(filepath)

    df = preprocess_data(df)
    
    if mode == "franchise":
        if start_date is None or end_date is None:
            start_date, end_date = get_date_range(df)
        df = filter_by_date_range(df, start_date, end_date)
        return df
    
    if partner_id is None:
        partner_id = df["partner_id"].iloc[0]
    
    df = filter_by_partner(df, partner_id)
    if start_date is None or end_date is None:
        start_date, end_date = get_date_range(df)
    
    return filter_by_date_range(df, start_date, end_date)
