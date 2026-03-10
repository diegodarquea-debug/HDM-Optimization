"""
Train predictive models for AWT and EPT.
Models: Linear Regression, Decision Tree, Random Forest
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .config import TRAIN_TEST_SPLIT, MODEL_TYPE, RANDOM_SEED, DEBUG


class AWTPredictor:
    """
    Predictive model for AWT (Avoidable Wait Time).
    Features: ordenes_pendientes, riders_cerca, hdm_activo
    Target: max_awt_espera_min (or awt_promedio if available)
    """
    
    def __init__(self, model_type=MODEL_TYPE):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = ["ordenes_pendientes", "riders_cerca", "hdm_activo"]
        self.ept_feature_name = None
        self.target_name = None
        self.metrics = {}
    
    def prepare_data(self, df):
        """
        Prepare X (features) and y (target) for training.
        
        Args:
            df: DataFrame with required columns
            
        Returns:
            X, y
        """
        # Prefer WIP-based EPT as explanatory feature when available
        if "ept_promedio_min_smoothed" in df.columns:
            self.ept_feature_name = "ept_promedio_min_smoothed"
        elif "ept_promedio_min" in df.columns:
            self.ept_feature_name = "ept_promedio_min"
        elif "ept_promedio" in df.columns:
            self.ept_feature_name = "ept_promedio"
        else:
            self.ept_feature_name = None

        if self.ept_feature_name:
            self.feature_names = ["ordenes_pendientes", "riders_cerca", "hdm_activo", self.ept_feature_name]
        else:
            self.feature_names = ["ordenes_pendientes", "riders_cerca", "hdm_activo"]

        X = df[self.feature_names].values
        
        # Use awt_promedio if available, else use max_awt_espera_min
        if "awt_promedio" in df.columns and df["awt_promedio"].mean() > 0:
            y = df["awt_promedio"].values
            self.target_name = "awt_promedio"
        else:
            y = df["max_awt_espera_min"].values
            self.target_name = "max_awt_espera_min"
        
        return X, y
    
    def train(self, df):
        """
        Train the AWT predictor.
        
        Args:
            df: Training DataFrame
        """
        X, y = self.prepare_data(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
        )
        
        # Initialize and train model
        if self.model_type == "linear_regression":
            self.model = LinearRegression()
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=5, 
                                               random_state=RANDOM_SEED, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        self.metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
        }
        
        if DEBUG:
            print(f"\n[MODEL] AWT Predictor ({self.model_type}) trained:")
            print(f"  Train RMSE: {self.metrics['train_rmse']:.3f}")
            print(f"  Test RMSE: {self.metrics['test_rmse']:.3f}")
            print(f"  Test MAE: {self.metrics['test_mae']:.3f}")
            print(f"  Test R²: {self.metrics['test_r2']:.3f}")
    
    def predict(self, ordenes_pendientes, riders_cerca, hdm_activo, ept_promedio_min=None):
        """
        Predict AWT given state variables.
        
        Args:
            ordenes_pendientes: Number of pending orders
            riders_cerca: Number of nearby riders
            hdm_activo: Whether HDM is active (0 or 1)
            
        Returns:
            Predicted AWT (minutes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.ept_feature_name:
            if ept_promedio_min is None:
                # Fallback conservative estimate when ept input is unavailable
                ept_promedio_min = max(0.0, float(ordenes_pendientes) * 0.5)
            X = np.array([[ordenes_pendientes, riders_cerca, hdm_activo, ept_promedio_min]])
        else:
            X = np.array([[ordenes_pendientes, riders_cerca, hdm_activo]])
        return float(self.model.predict(X)[0])
    
    def predict_batch(self, states):
        """
        Predict AWT for multiple states.
        
        Args:
            states: Array of shape (n_samples, 3) with columns
                   [ordenes_pendientes, riders_cerca, hdm_activo]
            
        Returns:
            Array of predicted AWT values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(states)


class EPTPredictor:
    """
    Predictive model for EPT (Estimated Prep Time).
    Features: ordenes_pendientes, riders_cerca, hdm_activo
    Target: ept_promedio (if available) or estimated from historical avg
    """
    
    def __init__(self, model_type=MODEL_TYPE):
        self.model_type = model_type
        self.model = None
        self.baseline_ept = None  # Baseline EPT when HDM is not active
        self.feature_names = ["ordenes_pendientes", "riders_cerca", "hdm_activo"]
        self.metrics = {}
    
    def prepare_data(self, df):
        """
        Prepare X (features) and y (target) for training.
        
        Args:
            df: DataFrame with required columns
            
        Returns:
            X, y (or None if EPT columns are not available)
        """
        # Priority order: WIP-based EPT first
        if "ept_promedio_min_smoothed" in df.columns and df["ept_promedio_min_smoothed"].mean() > 0:
            X = df[self.feature_names].values
            y = df["ept_promedio_min_smoothed"].values
            return X, y
        elif "ept_promedio_min" in df.columns and df["ept_promedio_min"].mean() > 0:
            X = df[self.feature_names].values
            y = df["ept_promedio_min"].values
            return X, y
        elif "ept_promedio" in df.columns and df["ept_promedio"].mean() > 0:
            X = df[self.feature_names].values
            y = df["ept_promedio"].values
            return X, y
        elif "ept_configurado_min" in df.columns and df["ept_configurado_min"].mean() > 0:
            X = df[self.feature_names].values
            y = df["ept_configurado_min"].values
            return X, y
        else:
            # Estimate EPT from historical baseline
            self.baseline_ept = df[df["hdm_activo"] == 0]["max_awt_espera_min"].mean() / 2
            return None, None
    
    def train(self, df):
        """
        Train the EPT predictor or set baseline.
        
        Args:
            df: Training DataFrame
        """
        X, y = self.prepare_data(df)
        
        if X is None or y is None:
            # No EPT data available, use baseline
            if DEBUG:
                print(f"\n[MODEL] No ept_promedio data found. Using baseline EPT: {self.baseline_ept:.2f} min")
            return
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
        )
        
        # Initialize and train model
        if self.model_type == "linear_regression":
            self.model = LinearRegression()
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=5,
                                               random_state=RANDOM_SEED, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_test = self.model.predict(X_test)
        
        self.metrics = {
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_r2": r2_score(y_test, y_pred_test),
        }
        
        if DEBUG:
            print(f"\n[MODEL] EPT Predictor ({self.model_type}) trained:")
            print(f"  Test RMSE: {self.metrics['test_rmse']:.3f}")
            print(f"  Test MAE: {self.metrics['test_mae']:.3f}")
            print(f"  Test R²: {self.metrics['test_r2']:.3f}")
    
    
    def predict(self, ordenes_pendientes, riders_cerca, hdm_activo):
        """
        Predict EPT given state variables.
        
        Args:
            ordenes_pendientes: Number of pending orders
            riders_cerca: Number of nearby riders
            hdm_activo: Whether HDM is active (0 or 1)
            
        Returns:
            Predicted EPT (minutes)
        """
        if self.model is None:
            if self.baseline_ept is not None:
                # Return baseline, optionally adjusted for HDM
                if hdm_activo:
                    return self.baseline_ept * 1.1  # Slight increase with HDM
                return self.baseline_ept
            raise ValueError("Model not trained. Call train() first.")
        
        X = np.array([[ordenes_pendientes, riders_cerca, hdm_activo]])
        return float(self.model.predict(X)[0])
    
    def predict_batch(self, states):
        """
        Predict EPT for multiple states.
        
        Args:
            states: Array of shape (n_samples, 3) with columns
                   [ordenes_pendientes, riders_cerca, hdm_activo]
            
        Returns:
            Array of predicted EPT values
        """
        if self.model is None:
            if self.baseline_ept is not None:
                predictions = np.full(len(states), self.baseline_ept)
                # Adjust for HDM where active
                predictions[states[:, 2] == 1] *= 1.1
                return predictions
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(states)


def train_models(df):
    """
    Train both AWT and EPT predictors.
    
    Args:
        df: Training DataFrame
        
    Returns:
        (awt_predictor, ept_predictor)
    """
    if DEBUG:
        print("\n" + "="*60)
        print("TRAINING PREDICTIVE MODELS")
        print("="*60)
    
    awt_pred = AWTPredictor(model_type=MODEL_TYPE)
    awt_pred.train(df)
    
    ept_pred = EPTPredictor(model_type=MODEL_TYPE)
    ept_pred.train(df)
    
    return awt_pred, ept_pred
