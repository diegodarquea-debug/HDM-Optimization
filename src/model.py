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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .config import TRAIN_TEST_SPLIT, MODEL_TYPE, RANDOM_SEED

logger = logging.getLogger(__name__)

class AWTPredictor:
    """Predictive model for AWT (Avoidable Wait Time)."""
    
    def __init__(self, model_type: str = MODEL_TYPE):
        self.model_type = model_type
        self.model = None
        self.feature_names = ["ordenes_pendientes", "riders_cerca", "hdm_activo"]
        self.ept_feature_name = None
        self.metrics = {}
    
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
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-TRAIN_TEST_SPLIT, random_state=RANDOM_SEED)
        
        if self.model_type == "linear_regression":
            self.model = LinearRegression()
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred),
        }
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
    
    def train(self, df: pd.DataFrame):
        """Train the EPT predictor."""
        ept_col = next((c for c in ["ept_promedio_min_smoothed", "ept_promedio_min", "ept_promedio", "ept_configurado_min"]
                        if c in df.columns and df[c].mean() > 0), None)
        
        if not ept_col:
            self.baseline_ept = df[df["hdm_activo"] == 0]["max_awt_espera_min"].mean() / 2
            logger.info(f"No EPT data. Using baseline: {self.baseline_ept:.2f}")
            return

        X = df[["ordenes_pendientes", "riders_cerca", "hdm_activo"]].values
        y = df[ept_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-TRAIN_TEST_SPLIT, random_state=RANDOM_SEED)
        
        if self.model_type == "linear_regression":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(random_state=RANDOM_SEED)
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED)
        else:
            raise ValueError(f"Unsupported EPT model_type: {self.model_type}")
        self.model.fit(X_train, y_train)
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
