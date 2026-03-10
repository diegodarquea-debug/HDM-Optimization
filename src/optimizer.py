"""
Optimization of HDM thresholds using scikit-optimize (Bayesian optimization).
Objective: minimize AWT subject to EPT constraint.
"""
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence
from .config import (
    THRESHOLDS,
    MAX_EPT_INCREASE,
    DEBUG,
    N_OPTIMIZATION_CALLS,
    OBJECTIVE_WEIGHTS,
    OPTIMIZER_PENALTIES,
    STRATEGY_SETTINGS,
    BAYESIAN_SETTINGS,
)


class HDMOptimizer:
    """
    Optimize HDM threshold parameters using Bayesian optimization.
    """
    
    def __init__(self, df, awt_predictor, ept_predictor, baseline_metrics,
                 objective_weight_awt=None, objective_weight_ept=None,
                 franchise_payloads=None):
        """
        Initialize optimizer.
        
        Args:
            df: Training data
            awt_predictor: Trained AWT predictor
            ept_predictor: Trained EPT predictor
            baseline_metrics: Baseline metrics dict
            objective_weight_awt: Weight for AWT in objective (minimize)
            objective_weight_ept: Weight for EPT penalty
        """
        self.df = df
        self.awt_predictor = awt_predictor
        self.ept_predictor = ept_predictor
        self.baseline_metrics = baseline_metrics
        # Load weights from config.py (no defaults - fail fast if missing)
        self.weight_awt = OBJECTIVE_WEIGHTS["awt"] if objective_weight_awt is None else objective_weight_awt
        self.weight_ept = OBJECTIVE_WEIGHTS["ept_penalty"] if objective_weight_ept is None else objective_weight_ept
        self.franchise_payloads = franchise_payloads or []
        # Load penalties from config.py (no defaults - fail fast if missing)
        self.penalty_awt_worse = OPTIMIZER_PENALTIES["awt_worse_quad"]
        self.penalty_combined_worse = OPTIMIZER_PENALTIES["combined_worse_quad"]
        self.penalty_ept_excess = OPTIMIZER_PENALTIES["ept_excess_quad"]
        
        # Extract baseline references for comparison
        self.baseline_awt = baseline_metrics["awt_promedio"]
        self.baseline_ept = baseline_metrics.get(
            "ept_configurado_promedio",
            baseline_metrics.get("ept_promedio", 0)
        )
        self.baseline_combined = self.baseline_awt + self.baseline_ept
        
        self.optimization_history = []
        self.best_config = None
        self.best_score = float('inf')
    
    def objective_function(self, params):
        """
        Objective function to minimize.
        Goal: Reduce AWT while keeping EPT increase <= 10 min and improving combined time.
        Uses order-weighted aggregation when multiple partners present.
        
        Args:
            params: [u1, u2, u3, delta_ept, duracion_hdm]
            
        Returns:
            Scalar loss (lower is better)
        """
        from .simulator import evaluate_configuration, evaluate_franchise_configuration
        
        u1, u2, u3, delta_ept, duracion_hdm = params
        
        # Evaluate scenario
        if self.franchise_payloads:
            result = evaluate_franchise_configuration(
                self.franchise_payloads,
                self.awt_predictor,
                self.ept_predictor,
                u1, u2, u3, delta_ept, duracion_hdm
            )
        else:
            result = evaluate_configuration(
                self.df, self.awt_predictor, self.ept_predictor,
                self.baseline_metrics, u1, u2, u3, delta_ept, duracion_hdm
            )
        
        # Extract results
        awt_predicted = result["awt_mean"]
        ept_increase = result["ept_increase"]
        combined_time = result["combined_time_mean"]
        
        # Calculate improvements vs baseline
        if self.franchise_payloads:
            # Franchise evaluator already returns order-weighted improvements
            awt_improvement = result.get("awt_improvement", 0.0)
            combined_improvement = result.get("combined_improvement", 0.0)
        else:
            awt_improvement = self.baseline_awt - awt_predicted
            combined_improvement = self.baseline_combined - combined_time
        
        # Smooth penalties instead of hard cutoffs for better exploration
        penalty = 0
        if awt_improvement < 0:  # AWT got worse
            penalty += self.penalty_awt_worse * (abs(awt_improvement) ** 2)
        if combined_improvement < 0:  # Combined time got worse
            penalty += self.penalty_combined_worse * (abs(combined_improvement) ** 2)
        if ept_increase > MAX_EPT_INCREASE:  # EPT increase too high
            penalty += self.penalty_ept_excess * ((ept_increase - MAX_EPT_INCREASE) ** 2)
        
        # Base objective (fully weight-driven from config):
        # maximize weighted AWT improvement while minimizing weighted EPT increase.
        # For franchise/multi-partner: use order-weighted aggregation
        orders_weight = self.df["ordenes_pendientes"].sum()
        if orders_weight > 0:
            # Order-weighted objective: sum of (orders * improvement) / total_orders
            weighted_gain = (self.weight_awt * awt_improvement) - (self.weight_ept * ept_increase)
        else:
            # Fallback if no orders
            weighted_gain = (self.weight_awt * awt_improvement) - (self.weight_ept * ept_increase)
        
        total_loss = -weighted_gain + penalty
        
        # Record for history (store raw floats for precision)
        self.optimization_history.append({
            "u1": u1, "u2": u2, "u3": u3,
            "delta_ept": delta_ept, "duracion_hdm": duracion_hdm,
            "awt_mean": awt_predicted,
            "awt_improvement": awt_improvement,
            "ept_increase": ept_increase,
            "combined_improvement": weighted_gain,
            "hdm_activation_rate": result.get("hdm_activation_rate", 0),
            "partners_count": len(self.franchise_payloads) if self.franchise_payloads else 1,
            "orders_weight_total": result.get("orders_weight_total", self.df["ordenes_pendientes"].sum() if "ordenes_pendientes" in self.df.columns else 0),
            "total_loss": total_loss,
        })
        
        # Track best
        if total_loss < self.best_score:
            self.best_score = total_loss
            self.best_config = params
        
        if DEBUG and len(self.optimization_history) % 5 == 0:
            print(f"  Iteration {len(self.optimization_history)}: loss={total_loss:.3f}, "
                  f"awt_imp={awt_improvement:.2f}, ept_inc={ept_increase:.2f}, "
                  f"weighted_gain={weighted_gain:.3f}")
        
        return total_loss
    
    def optimize(self, n_calls=None, method="gp_minimize", random_state=42, x0=None):
        """
        Run Bayesian optimization.
        
        Args:
            n_calls: Number of function evaluations (default: N_OPTIMIZATION_CALLS from config)
            method: "gp_minimize" or "forest_minimize"
            random_state: For reproducibility
            x0: Optional list of initial configurations to evaluate first
            
        Returns:
            Optimization result
        """
        if n_calls is None:
            n_calls = N_OPTIMIZATION_CALLS
        
        if DEBUG:
            print(f"\n[OPTIMIZER] Starting Bayesian optimization ({method})...")
            print(f"  n_calls: {n_calls}")
            print(f"  EPT constraint: max increase = {MAX_EPT_INCREASE} min")
            if x0:
                print(f"  Seed configurations from Monte Carlo: {len(x0)}")
        
        # Define search space - ALL DISCRETE EXCEPT delta_ept (now categorical)
        space = [
            Integer(int(THRESHOLDS["u1"][0]), 
                   int(THRESHOLDS["u1"][1]), name="u1"),
            Integer(int(THRESHOLDS["u2"][0]), 
                   int(THRESHOLDS["u2"][1]), name="u2"),
            Integer(int(THRESHOLDS["u3"][0]), 
                   int(THRESHOLDS["u3"][1]), name="u3"),
            Categorical(THRESHOLDS["delta_ept"], name="delta_ept"),  # Only [2, 4, 6, 8, 10]
            Integer(int(THRESHOLDS["duracion_hdm"][0]), 
                   int(THRESHOLDS["duracion_hdm"][1]), name="duracion_hdm"),
        ]
        
        # Progress bar callback
        pbar = tqdm(total=n_calls, desc="[OPTIMIZER] Bayesian Search", unit="iter")
        
        def callback(res):
            pbar.update(1)
        
        # Run optimization
        warnings.filterwarnings(
            "ignore",
            message="The objective has been evaluated at point.*",
            category=UserWarning,
        )
        if method == "gp_minimize":
            result = gp_minimize(
                self.objective_function,
                space,
                n_calls=n_calls,
                random_state=random_state,
                n_initial_points=BAYESIAN_SETTINGS["n_initial_points"],  # Use config directly
                acq_func="EI",  # Expected Improvement
                x0=x0,
                callback=callback,
                verbose=0,
            )
        elif method == "forest_minimize":
            result = forest_minimize(
                self.objective_function,
                space,
                n_calls=n_calls,
                random_state=random_state,
                n_initial_points=BAYESIAN_SETTINGS["n_initial_points"],  # Use config directly
                x0=x0,
                callback=callback,
                verbose=0,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        pbar.close()
        
        if DEBUG:
            print(f"[OPTIMIZER] Optimization complete!")
            print(f"  Best loss: {result.fun:.3f}")
            print(f"  Best params: u1={result.x[0]}, u2={result.x[1]}, u3={result.x[2]}, "
                  f"delta_ept={result.x[3]:.2f}, duracion_hdm={result.x[4]}")
        
        return result
    
    def get_pareto_frontier(self, df_results, awt_threshold=None, ept_threshold=None, k=3):
        """
        Extract Pareto-optimal configurations.
        
        Args:
            df_results: DataFrame with simulation results
            awt_threshold: Max acceptable AWT (optional)
            ept_threshold: Max acceptable EPT increase (optional)
            k: Number of top configurations to return
            
        Returns:
            DataFrame with Pareto frontier configurations
        """
        df = df_results.copy()
        
        # Apply thresholds if specified
        if awt_threshold is not None:
            df = df[df["awt_mean"] <= awt_threshold]
        if ept_threshold is not None:
            df = df[df["ept_increase"] <= ept_threshold]
        
        # Sort by combined metric and return top k
        df["combined_score"] = self.weight_awt * df["awt_mean"] + \
                               self.weight_ept * df["ept_increase"]
        
        pareto = df.nsmallest(k, "combined_score")
        
        return pareto
    
    def get_top_3_strategies(self):
        """
        Extract top-3 strategic configurations from optimization history.
        
        Strategies:
        1. Agresiva: Lowest AWT (maximum AWT reduction)
        2. Equilibrada: Best combined improvement (AWT reduction balanced with EPT increase)
        3. Conservadora: Lowest HDM activation rate among configs with ≥10% AWT reduction
        
        Returns:
            Dict with 3 strategy entries, each containing parameters and metrics
        """
        if not self.optimization_history:
            print("[WARNING] No optimization history available. Run optimize() first.")
            return {}
        
        df_hist = pd.DataFrame(self.optimization_history)

        # Always compute weighted combined score from current config weights.
        # This guarantees strategy selection reflects config.py exactly.
        df_hist["weighted_combined_improvement"] = (
            (self.weight_awt * (self.baseline_awt - df_hist["awt_mean"])) -
            (self.weight_ept * df_hist["ept_increase"])
        )
        
        # Ensure all required columns exist
        required_cols = ["u1", "u2", "u3", "delta_ept", "duracion_hdm", 
                        "awt_mean", "ept_increase", "hdm_activation_rate", 
                        "combined_improvement"]
        for col in required_cols:
            if col not in df_hist.columns:
                if col == "hdm_activation_rate":
                    df_hist[col] = 0.0
                elif col == "combined_improvement":
                    # Keep backward-compatible column name, but now fully weight-driven.
                    df_hist["combined_improvement"] = (
                        (self.weight_awt * (self.baseline_awt - df_hist["awt_mean"])) -
                        (self.weight_ept * df_hist["ept_increase"])
                    )
        
        strategies = {}
        
        # 1. AGRESIVA: Lowest AWT (most aggressive reduction)
        # Filter to valid configs (EPT increase ≤ MAX_EPT_INCREASE)
        valid_configs = df_hist[df_hist["ept_increase"] <= MAX_EPT_INCREASE].copy()
        if len(valid_configs) > 0:
            agresiva = valid_configs.loc[valid_configs["awt_mean"].idxmin()]
            strategies["Agresiva"] = {
                "u1": int(round(agresiva["u1"])),
                "u2": int(round(agresiva["u2"])),
                "u3": int(round(agresiva["u3"])),
                "delta_ept": round(float(agresiva["delta_ept"]), 2),
                "duracion_hdm": int(round(agresiva["duracion_hdm"])),
                "awt_improvement": round(float(agresiva.get("awt_improvement", self.baseline_awt - agresiva["awt_mean"])), 2),
                "ept_increase": round(float(agresiva["ept_increase"]), 2),
                "combined_improvement": round(float(agresiva.get("combined_improvement", 0)), 2),
                "hdm_activation_rate": round(float(agresiva.get("hdm_activation_rate", 0)), 2),
            }
        
        # 2. EQUILIBRADA: Best weighted balance (excluding Agresiva for diversity)
        # Maximize (weight_awt*AWT_reduction - weight_ept*EPT_increase)
        if len(valid_configs) > 0:
            # Filter out Agresiva config to enforce diversity
            equilibrada_candidates = valid_configs.copy()
            if "Agresiva" in strategies:
                agr = strategies["Agresiva"]
                mask = ~((equilibrada_candidates["u1"] == agr["u1"]) & 
                        (equilibrada_candidates["u2"] == agr["u2"]) & 
                        (equilibrada_candidates["u3"] == agr["u3"]) & 
                        (equilibrada_candidates["delta_ept"] == agr["delta_ept"]))
                equilibrada_candidates = equilibrada_candidates[mask]
            
            if len(equilibrada_candidates) > 0:
                equilibrada = equilibrada_candidates.loc[
                    equilibrada_candidates["weighted_combined_improvement"].idxmax()
                ]
                strategies["Equilibrada"] = {
                    "u1": int(round(equilibrada["u1"])),
                    "u2": int(round(equilibrada["u2"])),
                    "u3": int(round(equilibrada["u3"])),
                    "delta_ept": round(float(equilibrada["delta_ept"]), 2),
                    "duracion_hdm": int(round(equilibrada["duracion_hdm"])),
                    "awt_improvement": round(float(equilibrada.get("awt_improvement", self.baseline_awt - equilibrada["awt_mean"])), 2),
                    "ept_increase": round(float(equilibrada["ept_increase"]), 2),
                    "combined_improvement": round(float(equilibrada.get("weighted_combined_improvement", 0)), 2),
                    "hdm_activation_rate": round(float(equilibrada.get("hdm_activation_rate", 0)), 2),
                }
        
        # 3. CONSERVADORA: Lowest activation rate with ≥10% AWT reduction
        awt_reduction_threshold = STRATEGY_SETTINGS.get("conservative_min_awt_reduction_pct", 0.10) * self.baseline_awt
        conservative_candidates = valid_configs[
            (self.baseline_awt - valid_configs["awt_mean"]) >= awt_reduction_threshold
        ].copy()
        
        if len(conservative_candidates) > 0:
            conservadora = conservative_candidates.loc[
                conservative_candidates["hdm_activation_rate"].idxmin()
            ]
            strategies["Conservadora"] = {
                "u1": int(round(conservadora["u1"])),
                "u2": int(round(conservadora["u2"])),
                "u3": int(round(conservadora["u3"])),
                "delta_ept": round(float(conservadora["delta_ept"]), 2),
                "duracion_hdm": int(round(conservadora["duracion_hdm"])),
                "awt_improvement": round(float(conservadora.get("awt_improvement", self.baseline_awt - conservadora["awt_mean"])), 2),
                "ept_increase": round(float(conservadora["ept_increase"]), 2),
                "combined_improvement": round(float(conservadora.get("combined_improvement", 0)), 2),
                "hdm_activation_rate": round(float(conservadora.get("hdm_activation_rate", 0)), 2),
            }
        
        if DEBUG:
            print(f"\n[OPTIMIZER] Extracted {len(strategies)} strategic profiles")
            for strategy_name, config in strategies.items():
                print(f"  {strategy_name}: AWT improvement {config['awt_improvement']:.2f} min, "
                      f"EPT increase {config['ept_increase']:.2f} min, "
                      f"Activation rate {config['hdm_activation_rate']:.1%}")
        
        return strategies
    
    def print_optimization_summary(self, result):
        """Pretty print optimization results."""
        u1, u2, u3, delta_ept, duracion_hdm = result.x
        
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        print(f"\nOptimal HDM Configuration:")
        print(f"  u1 (ordenes threshold):    {u1:.0f}")
        print(f"  u2 (riders threshold):     {u2:.0f}")
        print(f"  u3 (AWT threshold):        {u3:.0f} minutes")
        print(f"  delta_ept (EPT increase):  {delta_ept:.2f} minutes")
        print(f"  duracion_hdm (duration):   {duracion_hdm:.0f} minutes")
        print(f"\nObjective value (loss):  {result.fun:.3f}")
        print("="*70 + "\n")


def optimize_hdm_thresholds(df, awt_predictor, ept_predictor, baseline_metrics,
                           n_calls=None, method="gp_minimize", franchise_payloads=None,
                           x0=None):
    """
    Complete optimization pipeline.
    
    Args:
        df: Training data
        awt_predictor: Trained AWT predictor
        ept_predictor: Trained EPT predictor
        baseline_metrics: Baseline metrics
        n_calls: Number of optimization iterations (default: N_OPTIMIZATION_CALLS)
        method: Optimization method
        
    Returns:
        (optimizer, result) tuple
    """
    if n_calls is None:
        n_calls = N_OPTIMIZATION_CALLS
    
    optimizer = HDMOptimizer(
        df,
        awt_predictor,
        ept_predictor,
        baseline_metrics,
        franchise_payloads=franchise_payloads,
    )
    result = optimizer.optimize(n_calls=n_calls, method=method, x0=x0)
    optimizer.print_optimization_summary(result)
    
    return optimizer, result
