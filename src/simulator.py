"""
Monte Carlo simulation for HDM activation scenarios.
10,000 iterations to evaluate different threshold configurations.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Union
from joblib import Parallel, delayed
from .config import (
    N_SIMULATIONS,
    RANDOM_SEED,
    MAX_EPT_INCREASE,
    DEBUG,
    ACTIVATION_DELAY_MINUTES,
    STRESS_WINDOW_ROLLING_SIZE,
    STRESS_WINDOW_HALF_SIZE,
    HDM_EFFECT_SETTINGS,
)


def _evaluate_partner_task(p, awt_predictor, ept_predictor, u1, u2, u3, delta_ept, duracion_hdm):
    """Helper for parallel evaluation of partners."""
    res = evaluate_configuration(p["df"], awt_predictor, ept_predictor, p["baseline_metrics"], u1, u2, u3, delta_ept, duracion_hdm)
    weight = max(1.0, float(p["df"]["ordenes_pendientes"].sum()))
    return res, weight, p.get("partner_id"), p.get("partner_name")


class HDMSimulator:
    """
    Simulate HDM activation and its impact on AWT and EPT.
    """
    
    def __init__(self, awt_predictor, ept_predictor, baseline_metrics: Dict[str, Any], random_seed: int = RANDOM_SEED):
        """
        Initialize simulator.
        
        Args:
            awt_predictor: Trained AWTPredictor
            ept_predictor: Trained EPTPredictor
            baseline_metrics: Dict with baseline metrics (ept_promedio, etc.)
            random_seed: For reproducibility
        """
        self.awt_predictor = awt_predictor
        self.ept_predictor = ept_predictor
        self.baseline_metrics = baseline_metrics
        self.baseline_ept_ref = baseline_metrics.get(
            "ept_promedio_min_smoothed_promedio",
            baseline_metrics.get("ept_promedio", baseline_metrics.get("ept_configurado_promedio", 0.0)),
        )
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def should_activate_hdm(self, ordenes_pendientes: float, riders_cerca: float, max_awt_espera_min: float,
                            u1: int, u2: int, u3: int) -> bool:
        """
        Determine if HDM should be activated based on thresholds.
        """
        return (ordenes_pendientes >= u1) and \
               (riders_cerca >= u2) and \
               (max_awt_espera_min >= u3)

    def _run_simulation_loop(self, df: pd.DataFrame, u1: int, u2: int, u3: int,
                             delta_ept: float, duracion_hdm: int) -> pd.DataFrame:
        """
        Core simulation engine optimized for speed.
        """
        u1, u2, u3 = int(round(u1)), int(round(u2)), int(round(u3))
        duracion_hdm = int(round(duracion_hdm))
        
        df_sim = df.copy().reset_index(drop=True)
        # Use unix timestamps (ns) for faster comparison.
        # Ensure we are in nanoseconds by explicit casting to datetime64[ns]
        momento_values_ns = df_sim["momento_exacto"].values.astype('datetime64[ns]').view(np.int64)
        ordenes_values = df_sim["ordenes_pendientes"].values
        riders_values = df_sim["riders_cerca"].values
        awt_values = df_sim["max_awt_espera_min"].values
        
        # Select best available EPT source
        ept_cols = ["ept_promedio_min_smoothed", "ept_promedio_min", "ept_promedio", "ept_configurado_min"]
        ept_original_values = np.zeros(len(df_sim), dtype=float)
        for col in ept_cols:
            if col in df_sim.columns:
                ept_original_values = df_sim[col].values
                break

        # Output arrays
        hdm_activated_sim = np.zeros(len(df_sim), dtype=np.int8)
        hdm_active_sim = np.zeros(len(df_sim), dtype=np.int8)
        hdm_in_delay_sim = np.zeros(len(df_sim), dtype=np.int8)
        awt_predicted = np.zeros(len(df_sim), dtype=float)
        ept_with_hdm = np.zeros(len(df_sim), dtype=float)
        
        # Precompute conditions
        u1_met = (ordenes_values >= u1)
        u2_met = (riders_values >= u2)
        u3_met = (awt_values >= u3)
        and_triggered = u1_met & u2_met & u3_met

        # Optimization: Pre-extract model weights if it's linear regression
        is_linear = hasattr(self.awt_predictor, 'model') and \
                    hasattr(self.awt_predictor.model, 'coef_') and \
                    hasattr(self.awt_predictor.model, 'intercept_')
        if is_linear:
            coefs = self.awt_predictor.model.coef_
            intercept = self.awt_predictor.model.intercept_
            has_ept_feat = self.awt_predictor.ept_feature_name is not None

        # Factor for HDM effect
        reduction_per_min = HDM_EFFECT_SETTINGS["awt_delta_ept_reduction_per_min"]
        max_total_reduction = HDM_EFFECT_SETTINGS["awt_delta_ept_max_reduction"]
        hdm_factor = max(1.0 - max_total_reduction, 1.0 - (reduction_per_min * delta_ept))

        activation_queue_start_ns = None
        hdm_end_time_ns = None
        delay_ns = int(ACTIVATION_DELAY_MINUTES * 60 * 1000 * 1000 * 1000)
        hdm_dur_ns = int(duracion_hdm * 60 * 1000 * 1000 * 1000)
        
        for i in range(len(df_sim)):
            curr_ns = momento_values_ns[i]
            is_triggered = and_triggered[i]
            
            hdm_currently_active = 0
            in_delay_period = 0
            
            if activation_queue_start_ns is not None:
                # We use a small epsilon (1s in ns) to avoid float/rounding issues if any
                if curr_ns < activation_queue_start_ns + delay_ns:
                    in_delay_period = 1
                    hdm_currently_active = 0
                elif hdm_end_time_ns is not None and curr_ns < hdm_end_time_ns:
                    in_delay_period = 0
                    hdm_currently_active = 1
                else:
                    activation_queue_start_ns = None
                    hdm_end_time_ns = None
                    in_delay_period = 0
                    hdm_currently_active = 0
            
            if is_triggered:
                if activation_queue_start_ns is None:
                    hdm_activated_sim[i] = 1
                    activation_queue_start_ns = curr_ns
                    hdm_end_time_ns = curr_ns + delay_ns + hdm_dur_ns
                    # Re-evaluate based on the new queue start
                    if curr_ns < activation_queue_start_ns + delay_ns:
                        in_delay_period = 1
                        hdm_currently_active = 0
                    else:
                        in_delay_period = 0
                        hdm_currently_active = 1
                elif hdm_currently_active == 1:
                    # Extend HDM duration if triggered while already active
                    hdm_end_time_ns = curr_ns + hdm_dur_ns

            hdm_active_sim[i] = hdm_currently_active
            hdm_in_delay_sim[i] = in_delay_period
            
            ord_i = ordenes_values[i]
            ept_base = max(0.0, float(ept_original_values[i])) if ord_i > 0 else 0.0
            ept_sim = ept_base + delta_ept if hdm_currently_active else ept_base
            
            if is_linear:
                # Fast manual calculation for linear models
                # features order: [ordenes, riders, hdm, ept (optional)]
                rid_i = riders_values[i]
                hdm_val = float(hdm_currently_active)
                awt_pred = intercept + ord_i * coefs[0] + rid_i * coefs[1] + hdm_val * coefs[2]
                if has_ept_feat:
                    awt_pred += ept_sim * coefs[3]
            else:
                awt_pred = self.awt_predictor.predict(float(ord_i), float(riders_values[i]),
                                                     float(hdm_currently_active), ept_sim)
            
            if hdm_currently_active and delta_ept > 0:
                awt_pred *= hdm_factor
            
            awt_predicted[i] = max(0.0, awt_pred)
            ept_with_hdm[i] = ept_sim

        df_sim["hdm_activated_sim"] = hdm_activated_sim
        df_sim["hdm_active_sim"] = hdm_active_sim
        df_sim["hdm_in_delay_sim"] = hdm_in_delay_sim
        df_sim["awt_predicted"] = awt_predicted
        df_sim["ept_with_hdm"] = ept_with_hdm
        df_sim["u1_condition"] = u1_met.astype(int)
        df_sim["u2_condition"] = u2_met.astype(int)
        df_sim["u3_condition"] = u3_met.astype(int)
        df_sim["all_conditions_met"] = and_triggered.astype(int)
        df_sim["ept_base"] = ept_original_values

        # print(f"DEBUG: and_triggered: {and_triggered}")
        # print(f"DEBUG: hdm_activated_sim: {hdm_activated_sim}")
        # print(f"DEBUG: hdm_in_delay_sim: {hdm_in_delay_sim}")
        # print(f"DEBUG: hdm_active_sim: {hdm_active_sim}")

        return df_sim

    def simulate_scenario(self, df: pd.DataFrame, u1: int, u2: int, u3: int,
                          delta_ept: float, duracion_hdm: int) -> Dict[str, Any]:
        """
        Simulate scenario and return aggregated results.
        """
        df_sim = self._run_simulation_loop(df, u1, u2, u3, delta_ept, duracion_hdm)

        awt_baseline = self.baseline_metrics.get("awt_promedio", 0.0)
        combined_baseline = awt_baseline + self.baseline_ept_ref
        
        awt_mean = df_sim["awt_predicted"].mean()
        ept_mean = df_sim["ept_with_hdm"].mean()
        combined_mean = (df_sim["awt_predicted"] + df_sim["ept_with_hdm"]).mean()
        
        return {
            "u1": int(round(u1)), "u2": int(round(u2)), "u3": int(round(u3)),
            "delta_ept": round(delta_ept, 2), "duracion_hdm": int(round(duracion_hdm)),
            "awt_mean": round(awt_mean, 2),
            "awt_improvement": round(awt_baseline - awt_mean, 2),
            "ept_increase": round(ept_mean - self.baseline_ept_ref, 2),
            "combined_time_mean": round(combined_mean, 2),
            "combined_improvement": round(combined_baseline - combined_mean, 2),
            "hdm_activations": int(df_sim["hdm_activated_sim"].sum()),
            "hdm_activation_rate": round(df_sim["hdm_active_sim"].mean(), 4),
            "awt_p50": round(df_sim["awt_predicted"].quantile(0.5), 2),
            "awt_p95": round(df_sim["awt_predicted"].quantile(0.95), 2),
            "ept_mean": round(ept_mean, 2),
            "activation_delay_applied": ACTIVATION_DELAY_MINUTES,
        }
    
    def run_simulations(self, df: pd.DataFrame, param_space: Dict[str, Any], n_sims: int = 100) -> pd.DataFrame:
        """
        Run multiple Monte Carlo simulations in parallel.
        """
        if DEBUG: print(f"\n[SIMULATOR] Running {n_sims} parallel simulations...")

        # Use a dedicated RNG to sample all parameters in the main process,
        # avoiding use of the global np.random inside Parallel workers.
        rng = np.random.default_rng(RANDOM_SEED)

        sim_params = []
        for _ in range(n_sims):
            u1 = rng.uniform(param_space["u1"][0], param_space["u1"][1])
            u2 = rng.uniform(param_space["u2"][0], param_space["u2"][1])
            u3 = rng.uniform(param_space["u3"][0], param_space["u3"][1])

            delta_spec = param_space["delta_ept"]
            if isinstance(delta_spec, (list, tuple)):
                delta_ept = float(rng.choice(delta_spec))
            else:
                delta_ept = rng.uniform(delta_spec[0], delta_spec[1])

            duracion_hdm = rng.uniform(param_space["duracion_hdm"][0], param_space["duracion_hdm"][1])

            sim_params.append((u1, u2, u3, delta_ept, duracion_hdm))

        results = Parallel(n_jobs=-1)(
            delayed(self.simulate_scenario)(
                df,
                u1,
                u2,
                u3,
                delta_ept,
                duracion_hdm,
            )
            for (u1, u2, u3, delta_ept, duracion_hdm) in tqdm(sim_params, desc="Simulating")
        )
        return pd.DataFrame(results)
    
    def generate_stress_day_analysis(self, df: pd.DataFrame, u1: int, u2: int, u3: int,
                                     delta_ept: float, duracion_hdm: int, output_dir: Union[str, Any]) -> str:
        """
        Analyze high-stress period.
        """
        from pathlib import Path
        
        df_analysis = df.copy()
        df_analysis['rolling_max'] = df_analysis['max_awt_espera_min'].rolling(
            window=STRESS_WINDOW_ROLLING_SIZE,
            min_periods=1,
        ).max()
        idx = df_analysis['rolling_max'].idxmax()
        
        start = max(0, idx - STRESS_WINDOW_HALF_SIZE)
        end = min(len(df), idx + STRESS_WINDOW_HALF_SIZE)
        
        df_stress = self._run_simulation_loop(df.iloc[start:end], u1, u2, u3, delta_ept, duracion_hdm)
        
        out_path = Path(output_dir) / "stress_day_validation.csv"
        df_out = df_stress.rename(columns={
            "max_awt_espera_min": "awt_real",
            "hdm_activo": "hdm_real",
            "hdm_active_sim": "hdm_simulated",
            "hdm_in_delay_sim": "hdm_in_delay",
            "ordenes_pendientes": "ordenes",
            "riders_cerca": "riders",
        })
        df_out.to_csv(out_path, index=False)
        
        awt_real = df.iloc[start:end]["max_awt_espera_min"].mean()
        awt_sim = df_stress["awt_predicted"].mean()
        
        summary = f"Stress Analysis: Real AWT={awt_real:.2f}, Sim AWT={awt_sim:.2f}, Improvement={awt_real-awt_sim:.2f}"
        print(summary)
        return summary

    def generate_full_timeline_validation(self, df: pd.DataFrame, u1: int, u2: int, u3: int,
                                          delta_ept: float, duracion_hdm: int, output_dir: Union[str, Any]) -> str:
        """
        Full dataset validation.
        """
        from pathlib import Path
        df_sim = self._run_simulation_loop(df, u1, u2, u3, delta_ept, duracion_hdm)
        df_out = df_sim.rename(columns={
            "max_awt_espera_min": "awt_real",
            "hdm_activo": "hdm_real",
            "hdm_active_sim": "hdm_simulated",
            "hdm_in_delay_sim": "hdm_in_delay",
            "ordenes_pendientes": "ordenes",
            "riders_cerca": "riders",
        })
        out_path = Path(output_dir) / "full_timeline_validation.csv"
        df_out.to_csv(out_path, index=False)
        return str(out_path)


def evaluate_configuration(df: pd.DataFrame, awt_predictor, ept_predictor, baseline_metrics: Dict[str, Any],
                          u1: int, u2: int, u3: int, delta_ept: float, duracion_hdm: int) -> Dict[str, Any]:
    return HDMSimulator(awt_predictor, ept_predictor, baseline_metrics).simulate_scenario(df, u1, u2, u3, delta_ept, duracion_hdm)


def evaluate_franchise_configuration(partner_payloads: List[Dict[str, Any]], awt_predictor, ept_predictor,
                                     u1: int, u2: int, u3: int, delta_ept: float, duracion_hdm: int) -> Dict[str, Any]:
    """
    Weighted evaluation across franchise in parallel.
    """
    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_partner_task)(p, awt_predictor, ept_predictor, u1, u2, u3, delta_ept, duracion_hdm)
        for p in partner_payloads
    )

    total_weight = 0.0
    agg_metrics = {k: 0.0 for k in ["awt_mean", "awt_improvement", "ept_increase", "combined_time_mean", "combined_improvement", "hdm_activation_rate"]}
    partner_results = []

    for res, weight, pid, pname in results:
        total_weight += weight
        for k in agg_metrics: agg_metrics[k] += res[k] * weight

        partner_results.append({
            "partner_id": pid,
            "partner_name": pname,
            "awt_mean": res["awt_mean"],
            "awt_improvement": res["awt_improvement"],
            "ept_increase": res["ept_increase"],
            "combined_improvement": res["combined_improvement"],
            "hdm_activation_rate": res["hdm_activation_rate"]
        })

    if total_weight > 0:
        for k in agg_metrics: agg_metrics[k] /= total_weight

    return {**agg_metrics, "orders_weight_total": total_weight, "partner_results": partner_results}
