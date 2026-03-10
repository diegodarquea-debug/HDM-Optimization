"""
Data loading and preprocessing from CSV (and BigQuery in future).
Auto-detects partner_id and date ranges.
"""
import pandas as pd
from pathlib import Path
from .config import RAW_DATA_PATH, DEBUG


def load_csv_data(filepath=None):
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file. If None, uses default RAW_DATA_PATH.
        
    Returns:
        DataFrame with columns: momento_exacto, partner_id, ordenes_pendientes, 
                                riders_cerca, hdm_activo, hdm_autor, max_awt_espera_min,
                                ept_promedio (if available), awt_promedio (if available)
    """
    if filepath is None:
        filepath = RAW_DATA_PATH
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Convert datetime column
    df["momento_exacto"] = pd.to_datetime(df["momento_exacto"])
    
    if DEBUG:
        print(f"Loaded {len(df)} rows from {filepath}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['momento_exacto'].min()} to {df['momento_exacto'].max()}")
    
    return df


def get_unique_partners(df):
    """Get unique partner IDs from dataframe."""
    return df["partner_id"].unique()


def get_date_range(df):
    """Get min and max dates from dataframe."""
    return df["momento_exacto"].min(), df["momento_exacto"].max()


def filter_by_partner(df, partner_id):
    """Filter dataframe by partner_id."""
    return df[df["partner_id"] == partner_id].copy()


def filter_by_date_range(df, start_date, end_date):
    """Filter dataframe by date range."""
    return df[(df["momento_exacto"] >= start_date) & 
              (df["momento_exacto"] <= end_date)].copy()


def preprocess_data(df):
    """
    Preprocess data: handle missing values, validate columns, etc.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Ensure required columns exist
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
    
    # If EPT and AWT columns exist, ensure they're numeric
    if "ept_promedio" in df.columns:
        df["ept_promedio"] = pd.to_numeric(df["ept_promedio"], errors="coerce").fillna(0)
    if "ept_promedio_min" in df.columns:
        df["ept_promedio_min"] = pd.to_numeric(df["ept_promedio_min"], errors="coerce").fillna(0)
    if "ept_configurado_min" in df.columns:
        df["ept_configurado_min"] = pd.to_numeric(df["ept_configurado_min"], errors="coerce").fillna(0)
    if "awt_promedio" in df.columns:
        df["awt_promedio"] = pd.to_numeric(df["awt_promedio"], errors="coerce").fillna(0)

    # NEW EPT source logic (WIP-based): prioritize ept_promedio_min
    # 1) Business rule: if no pending orders, EPT must be 0
    # 2) Outlier treatment: clip high spikes (p99 on active minutes)
    # 3) Smoothing: rolling mean (5-min) to reduce single-order distortion
    if "ept_promedio_min" in df.columns:
        df.loc[df["ordenes_pendientes"] == 0, "ept_promedio_min"] = 0.0

        active_mask = df["ordenes_pendientes"] > 0
        if active_mask.any():
            ept_p99 = df.loc[active_mask, "ept_promedio_min"].quantile(0.99)
            df.loc[active_mask, "ept_promedio_min"] = df.loc[active_mask, "ept_promedio_min"].clip(lower=0, upper=ept_p99)

        # Smoothing over short horizon (5-minute moving average)
        df["ept_promedio_min_smoothed"] = df["ept_promedio_min"].rolling(window=5, min_periods=1).mean()
        # Preserve zero-EPT business rule after smoothing
        df.loc[df["ordenes_pendientes"] == 0, "ept_promedio_min_smoothed"] = 0.0

        # Canonical EPT column used by legacy modules
        df["ept_promedio"] = df["ept_promedio_min_smoothed"]
    
    # Sort by datetime
    df = df.sort_values("momento_exacto").reset_index(drop=True)
    
    if DEBUG:
        print(f"\nData preprocessing complete:")
        print(f"Rows: {len(df)}")
        print(f"Partners: {df['partner_id'].nunique()}")
        print(f"Date range: {df['momento_exacto'].min()} to {df['momento_exacto'].max()}")
        print(f"Missing values:\n{df.isnull().sum()}")
    
    return df


def load_and_prepare_data(filepath=None, partner_id=None, start_date=None, end_date=None, mode="partner"):
    """
    Complete pipeline: load → preprocess → filter.
    
    Args:
        filepath: Path to CSV
        partner_id: Filter by this partner (optional, ignored if mode='franchise')
        start_date: Start date (optional)
        end_date: End date (optional)
        mode: "partner" (single) or "franchise" (all partners)
        
    Returns:
        Preprocessed and filtered DataFrame
    """
    df = load_csv_data(filepath)
    df = preprocess_data(df)
    
    # FRANCHISE mode: keep all partners
    if mode == "franchise":
        if start_date is None or end_date is None:
            start_date, end_date = get_date_range(df)
            if DEBUG:
                print(f"No date range specified, using: {start_date} to {end_date}")
        
        df = filter_by_date_range(df, start_date, end_date)
        
        if DEBUG:
            print(f"\nFinal dataset (FRANCHISE MODE):")
            print(f"Partners: {df['partner_id'].nunique()}")
            print(f"Rows: {len(df)}")
            print(f"Date range: {df['momento_exacto'].min()} to {df['momento_exacto'].max()}")
        
        return df
    
    # PARTNER mode: single partner
    # Auto-detect partner if not specified
    if partner_id is None:
        partner_id = df["partner_id"].iloc[0]
        if DEBUG:
            print(f"\nNo partner_id specified, using: {partner_id}")
    
    df = filter_by_partner(df, partner_id)
    
    # Auto-detect date range if not specified
    if start_date is None or end_date is None:
        start_date, end_date = get_date_range(df)
        if DEBUG:
            print(f"No date range specified, using: {start_date} to {end_date}")
    
    df = filter_by_date_range(df, start_date, end_date)
    
    if DEBUG:
        print(f"\nFinal dataset (PARTNER MODE):")
        print(f"Partner: {partner_id}")
        print(f"Rows: {len(df)}")
        print(f"Date range: {df['momento_exacto'].min()} to {df['momento_exacto'].max()}")
    
    return df
