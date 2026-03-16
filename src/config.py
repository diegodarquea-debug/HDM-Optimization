"""
Central configuration for the HDM optimization pipeline.

This file is intentionally verbose so business and technical users can
understand what each parameter controls and why default values were chosen.
"""
import os
import logging
from pathlib import Path

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _get_int_env(name: str, default: int) -> int:
    """Return integer env var value or fallback to default when missing/invalid."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def configure_logging(log_level=None, log_file="pipeline.log") -> None:
    """
    Configure application-wide logging.

    This function should be called explicitly by the application entrypoint
    to avoid configuring logging as a side effect of importing this module.

    Parameters
    ----------
    log_level : str | None
        Optional log level override (e.g., "DEBUG", "INFO"). If None, the
        LOG_LEVEL constant is used.
    log_file : str
        Path to the log file to write to (default: "pipeline.log").
    """
    level = (log_level or LOG_LEVEL).upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Project root folder (one level above /src)
PROJECT_ROOT = Path(__file__).parent.parent
# Input data folder (raw CSV files)
DATA_DIR = PROJECT_ROOT / "data"
# Output artifacts folder (all generated CSVs)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# Source code folder
SRC_DIR = PROJECT_ROOT / "src"

# Auto-create required folders if missing
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Default input dataset path
RAW_DATA_PATH = DATA_DIR / "raw_data.csv"

# -----------------------------------------------------------------------------
# SIMULATION
# -----------------------------------------------------------------------------
# Number of random Monte Carlo scenarios (Step 4).
# QUÉ HACE: Define cuántas configuraciones aleatorias se exploran antes de Bayesian.
# RECOMENDACIÓN: 200-500 para iteración rápida, 1000-2000 para producción.
N_SIMULATIONS = _get_int_env("N_SIMULATIONS", 2000)

# Stress analysis window (used for diagnostic slices in simulator reports)
STRESS_WINDOW_ROLLING_SIZE = 60
STRESS_WINDOW_HALF_SIZE = 30

# Fixed seed for reproducibility.
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# BAYESIAN OPTIMIZATION
# -----------------------------------------------------------------------------
# Number of objective-function evaluations in Step 5.
# RECOMENDACIÓN: 40-60 para iteración rápida, 80-120 para búsqueda profunda.
N_OPTIMIZATION_CALLS = _get_int_env("N_OPTIMIZATION_CALLS", 80)

# Bayesian internal settings
BAYESIAN_SETTINGS = {
    "n_initial_points": 5,
    "mc_seed_top_k": 20,
}

# -----------------------------------------------------------------------------
# SEARCH SPACE (what optimizer is allowed to try)
# -----------------------------------------------------------------------------
# ⚠️ IMPORTANTE: Lógica estricta AND.
# HDM se activa SOLO cuando u1 AND u2 AND u3 son TRUE simultáneamente.
# UPDATED for JupyterHub deployment (14-day training window)
THRESHOLDS = {
    # u1 = órdenes pendientes (umbral de activación).
    "u1": (3, 10),

    # u2 = riders cercanos (umbral de activación) - narrowed per P95 analysis.
    "u2": (1, 3),

    # u3 = espera máxima (umbral de activación, minutos) - earlier engagement.
    "u3": (5, 10),

    # delta_ept = minutos EXTRA de EPT mientras HDM está activo - higher thresholds only.
    "delta_ept": [4, 6, 8, 10],

    # duracion_hdm = duración del HDM por activación (minutos).
    "duracion_hdm": (10, 20),
}

# -----------------------------------------------------------------------------
# HDM IMPACT ADJUSTMENT (post-prediction calibration)
# -----------------------------------------------------------------------------
HDM_EFFECT_SETTINGS = {
    # Reducción proporcional de AWT por cada minuto de delta_ept.
    "awt_delta_ept_reduction_per_min": 0.03,  # 3.0%
    # Tope de reducción total de AWT por este ajuste (30% = 0.30).
    "awt_delta_ept_max_reduction": 0.30,
}

# -----------------------------------------------------------------------------
# OPTIMIZATION CONSTRAINTS & OBJECTIVE
# -----------------------------------------------------------------------------
# Hard safety cap: aumento MÁXIMO de EPT promedio permitido (minutos).
MAX_EPT_INCREASE = 15

# Activation delay before HDM impact is applied (minutes)
ACTIVATION_DELAY_MINUTES = 2

# Objective weights used by optimizer:
# score = (awt_weight * awt_improvement) - (ept_penalty * ept_increase)
# UPDATED for JupyterHub deployment: more aggressive on AWT reduction, more permissive on EPT increase
OBJECTIVE_WEIGHTS = {
    "awt": 2.5,
    "ept_penalty": 0.10,
}

# Smooth penalty strengths used in optimizer objective.
OPTIMIZER_PENALTIES = {
    "awt_worse_quad": 50,
    "combined_worse_quad": 30,
    "ept_excess_quad": 10,
    # Rate overactivation penalty: HDM active > hdm_rate_threshold of the time
    # loses operational meaning. Penalizes quadratically beyond the threshold.
    "hdm_rate_threshold": 0.25,
    "hdm_rate_excess_coeff": 20.0,
}

# Strategy extraction rules
STRATEGY_SETTINGS = {
    "conservative_min_awt_reduction_pct": 0.05,
}

# -----------------------------------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------------------------------
TRAIN_TEST_SPLIT = 0.6
# Options: "linear_regression", "random_forest", "decision_tree"
MODEL_TYPE = "random_forest"

# Hyperparameters shared by AWTPredictor and EPTPredictor.
# max_depth caps tree depth to prevent overfitting on short training windows.
MODEL_SETTINGS = {
    "rf_n_estimators": 100,
    "rf_max_depth": 5,
    "dt_max_depth": 5,
}

# -----------------------------------------------------------------------------
# OPERATOR DEFAULTS (Chile deployment)
# Override via env vars so nothing is hardcoded in argparse defaults.
# -----------------------------------------------------------------------------
DEFAULT_COUNTRY_ID = int(os.getenv("DEFAULT_COUNTRY_ID", "2"))
DEFAULT_ENTITY_ID = os.getenv("DEFAULT_ENTITY_ID", "PY_CL")

# -----------------------------------------------------------------------------
# BIGQUERY (JupyterHub: uses Default Application Credentials)
# -----------------------------------------------------------------------------
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", None))
BQ_DATASET = os.getenv("BQ_DATASET", None)
BQ_TABLE = os.getenv("BQ_TABLE", None)
BQ_LOCATION = os.getenv("BQ_LOCATION", "US")
BQ_TIMEOUT_SECONDS = int(os.getenv("BQ_TIMEOUT_SECONDS", "300"))
# Lookback window (days) used ONLY by the fallback default table query
# (when no --bq-query-file is provided). The franchise SQL file overrides this.
BQ_DEFAULT_LOOKBACK_DAYS = int(os.getenv("BQ_DEFAULT_LOOKBACK_DAYS", "30"))

# Data source mode: auto | csv | bigquery
DATA_SOURCE = os.getenv("HDM_DATA_SOURCE", "auto").lower()

DEBUG = (LOG_LEVEL == "DEBUG")
