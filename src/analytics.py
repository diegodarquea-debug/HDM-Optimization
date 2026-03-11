"""
Exploratory analysis and metrics calculation.
"""
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def calculate_baseline_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate baseline metrics (without HDM intervention).
    """
    metrics = {
        "ordenes_promedio": df["ordenes_pendientes"].mean(),
        "ordenes_std": df["ordenes_pendientes"].std(),
        "ordenes_p50": df["ordenes_pendientes"].quantile(0.5),
        "ordenes_p95": df["ordenes_pendientes"].quantile(0.95),
        "riders_promedio": df["riders_cerca"].mean(),
        "riders_std": df["riders_cerca"].std(),
        "riders_p50": df["riders_cerca"].quantile(0.5),
        "riders_p95": df["riders_cerca"].quantile(0.95),
        "awt_promedio": df["max_awt_espera_min"].mean(),
        "awt_std": df["max_awt_espera_min"].std(),
        "awt_p50": df["max_awt_espera_min"].quantile(0.5),
        "awt_p95": df["max_awt_espera_min"].quantile(0.95),
        "hdm_activaciones": df["hdm_activo"].sum(),
        "hdm_tasa": df["hdm_activo"].mean(),
    }
    
    # EPT related metrics
    ept_cols = {
        "ept_promedio_min_smoothed": ["ept_promedio_min_smoothed_promedio", "ept_promedio_min_smoothed_p95"],
        "ept_promedio_min": ["ept_promedio_min_promedio", "ept_promedio_min_p95"],
        "ept_promedio": ["ept_promedio", "ept_p95"],
        "ept_configurado_min": ["ept_configurado_promedio", "ept_configurado_p95"]
    }

    for col, (mean_name, p95_name) in ept_cols.items():
        if col in df.columns:
            metrics[mean_name] = df[col].mean()
            metrics[p95_name] = df[col].quantile(0.95)
    
    if "awt_promedio" in df.columns:
        metrics["awt_promedio_real"] = df["awt_promedio"].mean()
    
    return metrics


def log_baseline_metrics(metrics: Dict[str, Any]):
    """Log baseline metrics."""
    msg = "\n" + "="*60 + "\nBASELINE METRICS\n" + "="*60 + "\n"
    msg += f"Ordenes: Avg={metrics['ordenes_promedio']:.2f}, P50={metrics['ordenes_p50']:.1f}, P95={metrics['ordenes_p95']:.1f}\n"
    msg += f"Riders: Avg={metrics['riders_promedio']:.2f}, P50={metrics['riders_p50']:.1f}, P95={metrics['riders_p95']:.1f}\n"
    msg += f"AWT: Avg={metrics['awt_promedio']:.2f} min, P50={metrics['awt_p50']:.1f} min, P95={metrics['awt_p95']:.1f} min\n"
    msg += f"HDM: Total={metrics['hdm_activaciones']:.0f}, Rate={metrics['hdm_tasa']:.2%}\n"
    
    if "ept_promedio_min_smoothed_promedio" in metrics:
        msg += f"EPT WIP (Suavizado): Avg={metrics['ept_promedio_min_smoothed_promedio']:.2f} min, P95={metrics['ept_promedio_min_smoothed_p95']:.2f} min\n"
    
    msg += "="*60
    logger.info(msg)


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for key variables.
    """
    cols = ["ordenes_pendientes", "riders_cerca", "max_awt_espera_min", "hdm_activo"]
    optional_cols = ["ept_promedio", "ept_promedio_min", "ept_promedio_min_smoothed", "ept_configurado_min", "awt_promedio"]
    cols.extend([c for c in optional_cols if c in df.columns])
    return df[cols].corr()


def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive exploratory analysis.
    """
    logger.info("Starting exploratory analysis...")
    metrics = calculate_baseline_metrics(df)
    log_baseline_metrics(metrics)
    
    corr_matrix = calculate_correlations(df)
    logger.debug(f"\nCorrelation Matrix:\n{corr_matrix.round(3)}")
    
    return {
        "baseline_metrics": metrics,
        "correlation_matrix": corr_matrix,
    }
