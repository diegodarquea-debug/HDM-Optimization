"""
Optimization of HDM thresholds using scikit-optimize (Bayesian optimization).
Objective: minimize AWT subject to EPT constraint.
"""
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical

from .config import (
    THRESHOLDS,
    MAX_EPT_INCREASE,
    N_OPTIMIZATION_CALLS,
    OBJECTIVE_WEIGHTS,
    OPTIMIZER_PENALTIES,
    STRATEGY_SETTINGS,
    BAYESIAN_SETTINGS,
)

logger = logging.getLogger(__name__)

class HDMOptimizer:
    """
    Optimize HDM threshold parameters using Bayesian optimization.
    """
    
    def __init__(self, df: pd.DataFrame, awt_predictor: Any, ept_predictor: Any,
                 baseline_metrics: Dict[str, Any],
                 objective_weight_awt: Optional[float] = None,
                 objective_weight_ept: Optional[float] = None,
                 franchise_payloads: Optional[List[Dict[str, Any]]] = None):
        
        self.df = df
        self.awt_predictor = awt_predictor
        self.ept_predictor = ept_predictor
        self.baseline_metrics = baseline_metrics
        self.weight_awt = OBJECTIVE_WEIGHTS["awt"] if objective_weight_awt is None else objective_weight_awt
        self.weight_ept = OBJECTIVE_WEIGHTS["ept_penalty"] if objective_weight_ept is None else objective_weight_ept
        self.franchise_payloads = franchise_payloads or []

        self.penalty_awt_worse = OPTIMIZER_PENALTIES["awt_worse_quad"]
        self.penalty_combined_worse = OPTIMIZER_PENALTIES["combined_worse_quad"]
        self.penalty_ept_excess = OPTIMIZER_PENALTIES["ept_excess_quad"]
        
        self.baseline_awt = baseline_metrics["awt_promedio"]
        self.baseline_ept = baseline_metrics.get("ept_promedio", 0)
        self.baseline_combined = self.baseline_awt + self.baseline_ept
        
        self.optimization_history = []
        self.best_config = None
        self.best_score = float('inf')
    
    def objective_function(self, params: List[Any]) -> float:
        """Objective function to minimize."""
        from .simulator import evaluate_configuration, evaluate_franchise_configuration
        
        u1, u2, u3, delta_ept, duracion_hdm = params
        
        if self.franchise_payloads:
            result = evaluate_franchise_configuration(
                self.franchise_payloads, self.awt_predictor, self.ept_predictor,
                u1, u2, u3, delta_ept, duracion_hdm
            )
        else:
            result = evaluate_configuration(
                self.df, self.awt_predictor, self.ept_predictor,
                self.baseline_metrics, u1, u2, u3, delta_ept, duracion_hdm
            )
        
        awt_imp = result.get("awt_improvement", self.baseline_awt - result["awt_mean"])
        ept_inc = result["ept_increase"]
        comb_imp = result.get("combined_improvement", self.baseline_combined - result["combined_time_mean"])
        
        penalty = 0.0
        if awt_imp < 0: penalty += self.penalty_awt_worse * (abs(awt_imp) ** 2)
        if comb_imp < 0: penalty += self.penalty_combined_worse * (abs(comb_imp) ** 2)
        if ept_inc > MAX_EPT_INCREASE: penalty += self.penalty_ept_excess * ((ept_inc - MAX_EPT_INCREASE) ** 2)
        
        weighted_gain = (self.weight_awt * awt_imp) - (self.weight_ept * ept_inc)
        total_loss = -weighted_gain + penalty
        
        self.optimization_history.append({
            "u1": u1, "u2": u2, "u3": u3, "delta_ept": delta_ept, "duracion_hdm": duracion_hdm,
            "awt_mean": result["awt_mean"], "awt_improvement": awt_imp,
            "ept_increase": ept_inc, "combined_improvement": comb_imp,
            "objective_score": weighted_gain,
            "hdm_activation_rate": result.get("hdm_activation_rate", 0),
            "total_loss": total_loss,
        })
        
        if total_loss < self.best_score:
            self.best_score = total_loss
            self.best_config = params
        
        return float(total_loss)
    
    def optimize(self, n_calls: Optional[int] = None, method: str = "gp_minimize",
                 random_state: int = 42, x0: Optional[List[List[Any]]] = None) -> Any:
        """Run Bayesian optimization."""
        n_calls = n_calls or N_OPTIMIZATION_CALLS
        logger.info(f"Starting Bayesian optimization ({method}) with {n_calls} calls...")
        
        space = [
            Integer(int(THRESHOLDS["u1"][0]), int(THRESHOLDS["u1"][1]), name="u1"),
            Integer(int(THRESHOLDS["u2"][0]), int(THRESHOLDS["u2"][1]), name="u2"),
            Integer(int(THRESHOLDS["u3"][0]), int(THRESHOLDS["u3"][1]), name="u3"),
            Categorical(THRESHOLDS["delta_ept"], name="delta_ept"),
            Integer(int(THRESHOLDS["duracion_hdm"][0]), int(THRESHOLDS["duracion_hdm"][1]), name="duracion_hdm"),
        ]
        
        pbar = tqdm(total=n_calls, desc="[OPTIMIZER] Bayesian Search")
        def callback(res): pbar.update(1)
        
        warnings.filterwarnings("ignore", message="The objective has been evaluated at point.*")
        
        opt_args = {
            "func": self.objective_function,
            "dimensions": space,
            "n_calls": n_calls,
            "random_state": random_state,
            "n_initial_points": BAYESIAN_SETTINGS["n_initial_points"],
            "x0": x0,
            "callback": callback,
            "verbose": False
        }

        if method == "gp_minimize":
            result = gp_minimize(**opt_args, acq_func="EI")
        elif method == "forest_minimize":
            result = forest_minimize(**opt_args)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        pbar.close()
        logger.info(f"Optimization complete. Best loss: {result.fun:.3f}")
        return result
    
    def get_top_3_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Extract strategic profiles."""
        if not self.optimization_history: return {}
        
        df_hist = pd.DataFrame(self.optimization_history)
        valid = df_hist[df_hist["ept_increase"] <= MAX_EPT_INCREASE].copy()
        if valid.empty: valid = df_hist.copy()
        
        strategies = {}
        # Agresiva
        best_agr = valid.loc[valid["awt_improvement"].idxmax()]
        strategies["Agresiva"] = self._row_to_config(best_agr)
        
        # Equilibrada
        best_eq = valid.loc[valid["combined_improvement"].idxmax()]
        strategies["Equilibrada"] = self._row_to_config(best_eq)
        
        # Conservadora
        min_imp = STRATEGY_SETTINGS["conservative_min_awt_reduction_pct"] * self.baseline_awt
        cons_cands = valid[valid["awt_improvement"] >= min_imp]
        if not cons_cands.empty:
            best_cons = cons_cands.loc[cons_cands["hdm_activation_rate"].idxmin()]
            strategies["Conservadora"] = self._row_to_config(best_cons)
            
        return strategies

    def _row_to_config(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "u1": int(row["u1"]), "u2": int(row["u2"]), "u3": int(row["u3"]),
            "delta_ept": float(row["delta_ept"]), "duracion_hdm": int(row["duracion_hdm"]),
            "awt_improvement": float(row["awt_improvement"]),
            "ept_increase": float(row["ept_increase"]),
            "combined_improvement": float(row["combined_improvement"]),
            "hdm_activation_rate": float(row["hdm_activation_rate"]),
        }

def optimize_hdm_thresholds(df: pd.DataFrame, awt_predictor: Any, ept_predictor: Any,
                           baseline_metrics: Dict[str, Any], n_calls: Optional[int] = None,
                           method: str = "gp_minimize",
                           franchise_payloads: Optional[List[Dict[str, Any]]] = None,
                           x0: Optional[List[List[Any]]] = None) -> Tuple[HDMOptimizer, Any]:
    optimizer = HDMOptimizer(df, awt_predictor, ept_predictor, baseline_metrics, franchise_payloads=franchise_payloads)
    result = optimizer.optimize(n_calls=n_calls, method=method, x0=x0)
    return optimizer, result
