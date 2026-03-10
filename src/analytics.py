"""
Exploratory analysis and metrics calculation.
"""
import pandas as pd
import numpy as np
from .config import DEBUG


def calculate_baseline_metrics(df):
    """
    Calculate baseline metrics (without HDM intervention).
    
    Args:
        df: DataFrame with columns momento_exacto, ordenes_pendientes, riders_cerca,
                                    max_awt_espera_min, ept_promedio (optional), 
                                    awt_promedio (optional)
    
    Returns:
        Dict with baseline metrics
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
    
    # Add EPT and AWT if available (priority: WIP-based ept_promedio_min)
    if "ept_promedio_min_smoothed" in df.columns:
        metrics["ept_promedio_min_smoothed_promedio"] = df["ept_promedio_min_smoothed"].mean()
        metrics["ept_promedio_min_smoothed_p95"] = df["ept_promedio_min_smoothed"].quantile(0.95)
    if "ept_promedio_min" in df.columns:
        metrics["ept_promedio_min_promedio"] = df["ept_promedio_min"].mean()
        metrics["ept_promedio_min_p95"] = df["ept_promedio_min"].quantile(0.95)
    if "ept_promedio" in df.columns:
        metrics["ept_promedio"] = df["ept_promedio"].mean()
        metrics["ept_p95"] = df["ept_promedio"].quantile(0.95)
    if "ept_configurado_min" in df.columns:
        metrics["ept_configurado_promedio"] = df["ept_configurado_min"].mean()
        metrics["ept_configurado_p95"] = df["ept_configurado_min"].quantile(0.95)
    
    if "awt_promedio" in df.columns:
        metrics["awt_promedio_real"] = df["awt_promedio"].mean()
    
    return metrics


def print_baseline_metrics(metrics):
    """Pretty print baseline metrics."""
    print("\n" + "="*60)
    print("BASELINE METRICS")
    print("="*60)
    
    print("\nOrdenes Pendientes:")
    print(f"  Promedio: {metrics['ordenes_promedio']:.2f}")
    print(f"  Std Dev: {metrics['ordenes_std']:.2f}")
    print(f"  P50: {metrics['ordenes_p50']:.1f}")
    print(f"  P95: {metrics['ordenes_p95']:.1f}")
    
    print("\nRiders Cerca:")
    print(f"  Promedio: {metrics['riders_promedio']:.2f}")
    print(f"  Std Dev: {metrics['riders_std']:.2f}")
    print(f"  P50: {metrics['riders_p50']:.1f}")
    print(f"  P95: {metrics['riders_p95']:.1f}")
    
    print("\nAWT (Avoidable Wait Time):")
    print(f"  Promedio: {metrics['awt_promedio']:.2f} min")
    print(f"  Std Dev: {metrics['awt_std']:.2f}")
    print(f"  P50: {metrics['awt_p50']:.1f} min")
    print(f"  P95: {metrics['awt_p95']:.1f} min")
    
    print("\nHDM Activation:")
    print(f"  Total activations: {metrics['hdm_activaciones']:.0f}")
    print(f"  Activation rate: {metrics['hdm_tasa']:.2%}")
    
    if "ept_promedio_min_smoothed_promedio" in metrics:
        print(f"\nEPT WIP (Suavizado):")
        print(f"  Promedio: {metrics['ept_promedio_min_smoothed_promedio']:.2f} min")
        print(f"  P95: {metrics['ept_promedio_min_smoothed_p95']:.2f} min")

    if "ept_promedio_min_promedio" in metrics:
        print(f"\nEPT WIP (Original):")
        print(f"  Promedio: {metrics['ept_promedio_min_promedio']:.2f} min")
        print(f"  P95: {metrics['ept_promedio_min_p95']:.2f} min")

    if "ept_promedio" in metrics:
        print(f"\nEPT (Estimated Prep Time):")
        print(f"  Promedio: {metrics['ept_promedio']:.2f} min")
        print(f"  P95: {metrics['ept_p95']:.2f} min")
    
    if "ept_configurado_promedio" in metrics:
        print(f"\nEPT Configurado Real:")
        print(f"  Promedio: {metrics['ept_configurado_promedio']:.2f} min")
        print(f"  P95: {metrics['ept_configurado_p95']:.2f} min")
    
    print("="*60 + "\n")


def calculate_correlations(df):
    """
    Calculate correlation matrix for key variables.
    
    Args:
        df: DataFrame
        
    Returns:
        Correlation matrix
    """
    cols = ["ordenes_pendientes", "riders_cerca", "max_awt_espera_min", "hdm_activo"]
    
    # Add optional columns if they exist
    if "ept_promedio" in df.columns:
        cols.append("ept_promedio")
    if "ept_promedio_min" in df.columns:
        cols.append("ept_promedio_min")
    if "ept_promedio_min_smoothed" in df.columns:
        cols.append("ept_promedio_min_smoothed")
    if "ept_configurado_min" in df.columns:
        cols.append("ept_configurado_min")
    if "awt_promedio" in df.columns:
        cols.append("awt_promedio")
    
    return df[cols].corr()


def print_correlations(corr_matrix):
    """Pretty print correlation matrix."""
    print("\n" + "="*60)
    print("CORRELATION MATRIX")
    print("="*60)
    print(corr_matrix.round(3))
    print("="*60 + "\n")


def analyze_data(df):
    """
    Perform comprehensive exploratory analysis.
    
    Args:
        df: DataFrame
        
    Returns:
        Dict with analysis results
    """
    if DEBUG:
        print("\n[ANALYTICS] Starting exploratory analysis...")
    
    metrics = calculate_baseline_metrics(df)
    print_baseline_metrics(metrics)
    
    corr_matrix = calculate_correlations(df)
    print_correlations(corr_matrix)
    
    analysis = {
        "baseline_metrics": metrics,
        "correlation_matrix": corr_matrix,
    }
    
    return analysis
