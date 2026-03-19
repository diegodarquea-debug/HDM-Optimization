"""
Train predictive models for AWT and EPT.
"""
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .config import TRAIN_TEST_SPLIT, MODEL_TYPE, RANDOM_SEED, MODEL_SETTINGS

logger = logging.getLogger(__name__)


def sort_temporal_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe sorted chronologically for deterministic temporal splits."""
    if "momento_exacto" not in df.columns:
        return df.reset_index(drop=True).copy()

    sort_cols = ["momento_exacto"]
    if "partner_id" in df.columns:
        sort_cols.append("partner_id")

    return df.sort_values(sort_cols).reset_index(drop=True).copy()


def temporal_train_test_split_df(df: pd.DataFrame, train_ratio: float = TRAIN_TEST_SPLIT) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split a dataframe in chronological order (train first, test last).

    Returns
    -------
    (df_train, df_test, split_metadata)
    """
    if df.empty:
        return df.copy(), df.copy(), {
            "n_total": 0,
            "n_train": 0,
            "n_test": 0,
            "train_ratio": float(train_ratio),
            "test_ratio": 1.0 - float(train_ratio),
            "cutoff_timestamp": None,
            "first_test_timestamp": None,
            "is_temporal": True,
        }

    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1. Received: {train_ratio}")

    df_sorted = sort_temporal_dataframe(df)
    n_total = len(df_sorted)

    if n_total == 1:
        df_train = df_sorted.iloc[:1].copy()
        df_test = df_sorted.iloc[0:0].copy()
    else:
        n_train = int(np.floor(n_total * float(train_ratio)))
        n_train = max(1, min(n_total - 1, n_train))
        df_train = df_sorted.iloc[:n_train].copy()
        df_test = df_sorted.iloc[n_train:].copy()

    cutoff_timestamp = None
    first_test_timestamp = None
    if "momento_exacto" in df_train.columns and not df_train.empty:
        cutoff_timestamp = df_train["momento_exacto"].iloc[-1]
    if "momento_exacto" in df_test.columns and not df_test.empty:
        first_test_timestamp = df_test["momento_exacto"].iloc[0]

    split_metadata = {
        "n_total": n_total,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "train_ratio": float(train_ratio),
        "test_ratio": 1.0 - float(train_ratio),
        "cutoff_timestamp": cutoff_timestamp,
        "first_test_timestamp": first_test_timestamp,
        "is_temporal": True,
    }
    return df_train, df_test, split_metadata

class AWTPredictor:
    """Predictive model for AWT (Avoidable Wait Time)."""
    
    def __init__(self, model_type: str = MODEL_TYPE):
        self.model_type = model_type
        self.model = None
        self.feature_names = ["ordenes_pendientes", "riders_cerca", "hdm_activo"]
        self.ept_feature_name = None
        self.metrics = {}
        self.split_metadata: Dict[str, Any] = {}
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare X and y for training."""
        ept_candidates = ["ept_promedio_min_smoothed", "ept_promedio_min", "ept_promedio"]
        self.ept_feature_name = next((c for c in ept_candidates if c in df.columns), None)

        features = ["ordenes_pendientes", "riders_cerca", "hdm_activo"]
        if self.ept_feature_name:
            features.append(self.ept_feature_name)
        
        self.feature_names = features
        X = df[features].values
        y = df["awt_promedio" if "awt_promedio" in df.columns and df["awt_promedio"].mean() > 0 else "max_awt_espera_min"].values
        return X, y
    
    def train(self, df: pd.DataFrame):
        """Train the AWT predictor."""
        df_train, df_test, split_metadata = temporal_train_test_split_df(df, TRAIN_TEST_SPLIT)

        X_train, y_train = self.prepare_data(df_train)
        X_test, y_test = self.prepare_data(df_test) if not df_test.empty else (None, None)
        
        if self.model_type == "linear_regression":
            self.model = LinearRegression()
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeRegressor(max_depth=MODEL_SETTINGS["dt_max_depth"], random_state=RANDOM_SEED)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=MODEL_SETTINGS["rf_n_estimators"], max_depth=MODEL_SETTINGS["rf_max_depth"], random_state=RANDOM_SEED, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)

        if X_test is not None and len(X_test) > 0:
            y_pred = self.model.predict(X_test)
            self.metrics = {
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "test_mae": mean_absolute_error(y_test, y_pred),
                "test_r2": r2_score(y_test, y_pred),
            }
        else:
            self.metrics = {
                "test_rmse": 0.0,
                "test_mae": 0.0,
                "test_r2": 0.0,
            }

        self.split_metadata = split_metadata
        logger.info(f"AWT Predictor ({self.model_type}) trained. R2: {self.metrics['test_r2']:.3f}")
    
    def predict(self, ordenes: float, riders: float, hdm: float, ept: Optional[float] = None) -> float:
        """Predict AWT."""
        if self.model is None: raise ValueError("Model not trained.")
        
        inputs = [ordenes, riders, hdm]
        if self.ept_feature_name:
            inputs.append(ept if ept is not None else max(0.0, ordenes * 0.5))
        
        return float(self.model.predict(np.array([inputs]))[0])


class EPTPredictor:
    """Predictive model for EPT (Estimated Prep Time)."""
    
    def __init__(self, model_type: str = MODEL_TYPE):
        self.model_type = model_type
        self.model = None
        self.baseline_ept = None
        self.metrics = {}
        self.split_metadata: Dict[str, Any] = {}
    
    def train(self, df: pd.DataFrame):
        """Train the EPT predictor."""
        df_train, df_test, split_metadata = temporal_train_test_split_df(df, TRAIN_TEST_SPLIT)

        ept_col = next((c for c in ["ept_promedio_min_smoothed", "ept_promedio_min", "ept_promedio", "ept_configurado_min"]
                        if c in df_train.columns and df_train[c].mean() > 0), None)
        
        if not ept_col:
            self.baseline_ept = df_train[df_train["hdm_activo"] == 0]["max_awt_espera_min"].mean() / 2
            self.split_metadata = split_metadata
            logger.info(f"No EPT data. Using baseline: {self.baseline_ept:.2f}")
            return

        X_train = df_train[["ordenes_pendientes", "riders_cerca", "hdm_activo"]].values
        y_train = df_train[ept_col].values
        X_test = df_test[["ordenes_pendientes", "riders_cerca", "hdm_activo"]].values if not df_test.empty else None
        y_test = df_test[ept_col].values if not df_test.empty else None
        
        if self.model_type == "linear_regression":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=MODEL_SETTINGS["rf_n_estimators"], max_depth=MODEL_SETTINGS["rf_max_depth"], random_state=RANDOM_SEED)
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeRegressor(max_depth=MODEL_SETTINGS["dt_max_depth"], random_state=RANDOM_SEED)
        else:
            raise ValueError(f"Unsupported EPT model_type: {self.model_type}")
        self.model.fit(X_train, y_train)

        if X_test is not None and len(X_test) > 0:
            y_pred = self.model.predict(X_test)
            self.metrics = {
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "test_mae": mean_absolute_error(y_test, y_pred),
                "test_r2": r2_score(y_test, y_pred),
            }
        else:
            self.metrics = {
                "test_rmse": 0.0,
                "test_mae": 0.0,
                "test_r2": 0.0,
            }

        self.split_metadata = split_metadata
        logger.info(f"EPT Predictor trained on {ept_col}.")

    def predict(self, ordenes: float, riders: float, hdm: float) -> float:
        if self.model: return float(self.model.predict(np.array([[ordenes, riders, hdm]]))[0])
        return self.baseline_ept * (1.1 if hdm else 1.0) if self.baseline_ept else 0.0


def train_models(df: pd.DataFrame) -> Tuple[AWTPredictor, EPTPredictor]:
    """Train both predictors."""
    logger.info("Training predictive models...")
    awt_pred = AWTPredictor(); awt_pred.train(df)
    ept_pred = EPTPredictor(); ept_pred.train(df)
    return awt_pred, ept_pred
