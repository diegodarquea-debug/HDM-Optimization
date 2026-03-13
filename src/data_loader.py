"""
Data loading and preprocessing from CSV (and BigQuery in future).
Auto-detects partner_id and date ranges.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union, Any
import pandas as pd
from .config import RAW_DATA_PATH, GCP_PROJECT_ID, BQ_DATASET, BQ_TABLE

logger = logging.getLogger(__name__)

def load_bigquery_data(query: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from Google BigQuery.
    Requires GOOGLE_APPLICATION_CREDENTIALS to be set.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        logger.error("google-cloud-bigquery not installed.")
        raise ImportError("Please install google-cloud-bigquery to use BigQuery loading.")

    if not GCP_PROJECT_ID or not BQ_DATASET or not BQ_TABLE:
        logger.warning("BigQuery config missing in config.py. Falling back to default query.")
        # We can't really run a query without a project, but we'll try if a query is provided
        if not query:
            raise ValueError("BigQuery configuration missing and no query provided.")

    client = bigquery.Client(project=GCP_PROJECT_ID)

    if not query:
        query = f"""
            SELECT *
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
            WHERE momento_exacto >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        """

    logger.info(f"Executing BigQuery query: {query[:100]}...")
    df = client.query(query).to_dataframe()
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
                          source: str = "csv") -> pd.DataFrame:
    """Complete pipeline: load → preprocess → filter."""
    if source == "bigquery":
        df = load_bigquery_data()
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
