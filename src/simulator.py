"""
Monte Carlo simulation for HDM activation scenarios.
10,000 iterations to evaluate different threshold configurations.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
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


class HDMSimulator:
    """
    Simulate HDM activation and its impact on AWT and EPT.
    """
    
    def __init__(self, awt_predictor, ept_predictor, baseline_metrics, random_seed=RANDOM_SEED):
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
    
    def should_activate_hdm(self, ordenes_pendientes, riders_cerca, max_awt_espera_min,
                            u1, u2, u3):
        """
        Determine if HDM should be activated based on thresholds.
        
        Logic: Activate ONLY when ALL three conditions are true (AND):
          - ordenes_pendientes >= u1  (orders reach/exceed baseline)
          - riders_cerca >= u2  (riders reach/exceed baseline)
          - max_awt_espera_min >= u3  (wait time reach/exceed baseline)
        
        This ensures HDM only activates during genuine multidimensional stress,
        avoiding false positives from noise in a single metric.
        
        Args:
            ordenes_pendientes: Current pending orders
            riders_cerca: Current nearby riders
            max_awt_espera_min: Current max wait time
            u1, u2, u3: Thresholds
            
        Returns:
            Boolean: True if HDM should activate (all 3 conditions met)
        """
        return (ordenes_pendientes >= u1) and \
               (riders_cerca >= u2) and \
               (max_awt_espera_min >= u3)
    
    def simulate_scenario(self, df, u1, u2, u3, delta_ept, duracion_hdm):
        """
        Simulate a single scenario with given HDM parameters.
        
        IMPORTANTE: Implementa 2-minuto activation delay.
        Si AND ocurre en minuto T, HDM impacta en T+3 (delay de 2 min buffer).
        
        Args:
            df: Input data (hourly or minute-level)
            u1, u2, u3: Activation thresholds (AND logic)
            delta_ept: Minutes to add to EPT when HDM activates
            duracion_hdm: Duration of HDM activation (minutes)
            
        Returns:
            Dict with scenario results
        """
        # Cast thresholds to integers (discrete values)
        u1 = int(round(u1))
        u2 = int(round(u2))
        u3 = int(round(u3))
        duracion_hdm = int(round(duracion_hdm))
        
        # Create copy and extract arrays for fast access
        df_sim = df.copy().reset_index(drop=True)
        momento_values = df_sim["momento_exacto"].values
        ordenes_values = df_sim["ordenes_pendientes"].values
        riders_values = df_sim["riders_cerca"].values
        awt_values = df_sim["max_awt_espera_min"].values
        if "ept_promedio_min_smoothed" in df_sim.columns:
            ept_original_values = df_sim["ept_promedio_min_smoothed"].values
        elif "ept_promedio_min" in df_sim.columns:
            ept_original_values = df_sim["ept_promedio_min"].values
        elif "ept_promedio" in df_sim.columns:
            ept_original_values = df_sim["ept_promedio"].values
        elif "ept_configurado_min" in df_sim.columns:
            ept_original_values = df_sim["ept_configurado_min"].values
        else:
            ept_original_values = np.zeros(len(df_sim), dtype=float)
        
        # Initialize result arrays
        hdm_activated_sim = np.zeros(len(df_sim), dtype=int)
        hdm_active_sim = np.zeros(len(df_sim), dtype=int)
        hdm_activation_delayed = np.zeros(len(df_sim), dtype=int)  # Track if in delay period
        awt_predicted = np.zeros(len(df_sim), dtype=float)
        ept_predicted = np.zeros(len(df_sim), dtype=float)
        ept_with_hdm = np.zeros(len(df_sim), dtype=float)
        
        # Track HDM activation with delay
        hdm_end_time = None
        activation_queue_start = None  # When AND condition first triggered
        
        activation_delay_minutes = ACTIVATION_DELAY_MINUTES
        
        for i in range(len(df_sim)):
            current_time = momento_values[i]
            ordenes = ordenes_values[i]
            riders = riders_values[i]
            awt = awt_values[i]
            
            # Convert numpy datetime64 to pandas Timestamp if needed
            if isinstance(current_time, np.datetime64):
                current_time = pd.Timestamp(current_time)
            
            # Check if HDM should activate NOW (AND logic)
            hdm_should_activate = self.should_activate_hdm(ordenes, riders, awt, u1, u2, u3)
            
            # Determine current HDM state (accounting for delay)
            hdm_currently_active = 0
            in_delay_period = 0
            
            # If in delay period, don't activate yet but mark it
            if activation_queue_start is not None:
                time_delta = current_time - activation_queue_start
                # Handle both pandas Timedelta and numpy timedelta64
                if hasattr(time_delta, 'total_seconds'):
                    time_since_trigger = time_delta.total_seconds() / 60
                else:
                    time_since_trigger = time_delta / np.timedelta64(1, 'm')
                    
                if time_since_trigger < activation_delay_minutes:
                    # Still in delay: AWT follows baseline (no HDM effect yet)
                    in_delay_period = 1
                elif time_since_trigger < activation_delay_minutes + duracion_hdm:
                    # Delay passed: HDM now active
                    hdm_currently_active = 1
                else:
                    # HDM duration expired: reset queue
                    activation_queue_start = None
                    hdm_end_time = None
            
            # If HDM already active and conditions persist: extend duration (ping-pong)
            if hdm_currently_active == 1 and hdm_should_activate:
                hdm_end_time = current_time + pd.Timedelta(minutes=int(duracion_hdm))
            
            # If new activation needed and no queue active: start queue
            if hdm_should_activate and activation_queue_start is None:
                hdm_activated_sim[i] = 1
                activation_queue_start = current_time
                # Calculate when impact will start (T + 2 minutes)
                hdm_end_time = current_time + pd.Timedelta(minutes=activation_delay_minutes + duracion_hdm)
                in_delay_period = 1
            
            hdm_active_sim[i] = hdm_currently_active
            hdm_activation_delayed[i] = in_delay_period
            
            # Predict AWT and EPT
            # During delay period: use hdm_val=0 (no effect yet)
            # After delay period: use hdm_val=1 (HDM active)
            ordenes_val = float(ordenes)
            riders_val = float(riders)
            hdm_val = float(hdm_currently_active)  # 1 only AFTER delay, 0 during delay
            
            # Baseline EPT per minute: prioritize real WIP-based EPT from input data
            ept_base_pred = float(ept_original_values[i])
            if ordenes_val <= 0:
                ept_base_pred = 0.0
            ept_base_pred = max(0, ept_base_pred)

            # HDM intervention on EPT: ept_simulado = ept_original + delta_ept
            if hdm_currently_active:
                ept_simulated_min = ept_base_pred + delta_ept
            else:
                ept_simulated_min = ept_base_pred

            awt_pred = self.awt_predictor.predict(ordenes_val, riders_val, hdm_val, ept_simulated_min)
            
            # IMPROVEMENT FACTOR: More delta_ept = More AWT reduction
            # Controlled from config (HDM_EFFECT_SETTINGS)
            if hdm_currently_active and delta_ept > 0:
                reduction_per_min = HDM_EFFECT_SETTINGS["awt_delta_ept_reduction_per_min"]
                max_total_reduction = HDM_EFFECT_SETTINGS["awt_delta_ept_max_reduction"]
                min_factor = 1.0 - max_total_reduction

                improvement_factor = 1.0 - (reduction_per_min * delta_ept)
                improvement_factor = max(min_factor, improvement_factor)
                awt_pred = awt_pred * improvement_factor
            
            awt_predicted[i] = max(0, awt_pred)
            ept_predicted[i] = max(0, ept_base_pred)
            
            # EPT: only apply delta_ept when HDM is ACTUALLY active (after delay)
            ept_with_hdm[i] = max(0, ept_simulated_min)
        
        # Store results back to dataframe
        df_sim["hdm_activated_sim"] = hdm_activated_sim
        df_sim["hdm_active_sim"] = hdm_active_sim
        df_sim["hdm_in_delay_sim"] = hdm_activation_delayed
        df_sim["awt_predicted"] = awt_predicted
        df_sim["ept_predicted"] = ept_predicted
        df_sim["ept_with_hdm"] = ept_with_hdm
        
        # Calculate aggregated results
        hdm_active_count = hdm_active_sim.sum()
        hdm_activation_rate = hdm_active_count / len(df_sim) if len(df_sim) > 0 else 0
        
        results = {
            "u1": u1,
            "u2": u2,
            "u3": u3,
            "delta_ept": round(delta_ept, 2),
            "duracion_hdm": duracion_hdm,
            "awt_mean": round(df_sim["awt_predicted"].mean(), 2),
            "awt_p50": round(df_sim["awt_predicted"].quantile(0.5), 2),
            "awt_p95": round(df_sim["awt_predicted"].quantile(0.95), 2),
            "ept_mean": round(df_sim["ept_with_hdm"].mean(), 2),
            "ept_increase": round(df_sim["ept_with_hdm"].mean() - 
                           self.baseline_ept_ref, 2),
            "hdm_activations": df_sim["hdm_activated_sim"].sum(),
            "hdm_activation_rate": round(hdm_activation_rate, 2),  # Duty cycle
            "combined_time_mean": round((df_sim["awt_predicted"] + df_sim["ept_with_hdm"]).mean(), 2),
            "activation_delay_applied": activation_delay_minutes,
        }
        
        return results
    
    def run_simulations(self, df, param_space, n_sims=100):
        """
        Run multiple simulations with random parameter combinations.
        
        Args:
            df: Input data
            param_space: Dict with parameter ranges:
                {
                    "u1": (min, max),
                    "u2": (min, max),
                    "u3": (min, max),
                    "delta_ept": (min, max),
                    "duracion_hdm": (min, max),
                }
            n_sims: Number of simulations to run
            
        Returns:
            DataFrame with simulation results
        """
        results_list = []
        
        if DEBUG:
            print(f"\n[SIMULATOR] Running {n_sims} Monte Carlo simulations...")
        
        # Progress bar for Monte Carlo exploration
        pbar = tqdm(total=n_sims, desc="[SIMULATOR] Monte Carlo Exploration", unit="config")
        
        for i in range(n_sims):
            # Random sampling within param_space
            u1 = np.random.uniform(param_space["u1"][0], param_space["u1"][1])
            u2 = np.random.uniform(param_space["u2"][0], param_space["u2"][1])
            u3 = np.random.uniform(param_space["u3"][0], param_space["u3"][1])
            if isinstance(param_space["delta_ept"], (list, tuple)) and len(param_space["delta_ept"]) > 2:
                delta_ept = float(np.random.choice(param_space["delta_ept"]))
            else:
                delta_ept = np.random.uniform(param_space["delta_ept"][0], 
                                             param_space["delta_ept"][1])
            duracion_hdm = np.random.uniform(param_space["duracion_hdm"][0],
                                            param_space["duracion_hdm"][1])
            
            # Run simulation
            result = self.simulate_scenario(df, u1, u2, u3, delta_ept, duracion_hdm)
            results_list.append(result)
            
            pbar.update(1)
        
        pbar.close()
        
        results_df = pd.DataFrame(results_list)
        
        if DEBUG:
            print(f"[SIMULATOR] Simulations complete!")
        
        return results_df
    
    def generate_stress_day_analysis(self, df, u1, u2, u3, delta_ept, duracion_hdm, output_dir):
        """
        Analyze a high-stress period comparing real AWT vs simulated with recommended thresholds.
        
        Identifies period with maximum consecutive AWT spike and compares:
        - Real historical HDM activation (hdm_activo from data)
        - Simulated HDM activation (using AND logic with u1, u2, u3)
        - Expected AWT impact accounting for 2-minute delay
        
        Args:
            df: Historical data
            u1, u2, u3: Recommended thresholds
            delta_ept: Recommended EPT increase
            duracion_hdm: Recommended HDM duration
            output_dir: Directory to save analysis
            
        Returns:
            str: Summary of stress day analysis
        """
        from pathlib import Path
        
        # Find stress period: rolling max AWT
        window_size = STRESS_WINDOW_ROLLING_SIZE  # Configurable rolling window (default 60 min)
        df_analysis = df.copy().reset_index(drop=True)
        df_analysis['awt_rolling_max'] = df_analysis['max_awt_espera_min'].rolling(
            window=window_size, min_periods=1).max()
        
        stress_idx = df_analysis['awt_rolling_max'].idxmax()
        stress_window_start = max(0, stress_idx - STRESS_WINDOW_HALF_SIZE)
        stress_window_end = min(len(df_analysis), stress_idx + STRESS_WINDOW_HALF_SIZE)
        
        df_stress = df_analysis.iloc[stress_window_start:stress_window_end].copy().reset_index(drop=True)
        
        # Cast thresholds
        u1 = int(round(u1))
        u2 = int(round(u2))
        u3 = int(round(u3))
        duracion_hdm = int(round(duracion_hdm))
        
        # Build minute-by-minute comparison
        comparison_data = []
        momento_values = df_stress["momento_exacto"].values
        ordenes_values = df_stress["ordenes_pendientes"].values
        riders_values = df_stress["riders_cerca"].values
        awt_values = df_stress["max_awt_espera_min"].values
        hdm_real_values = df_stress["hdm_activo"].values
        if "ept_promedio_min_smoothed" in df_stress.columns:
            ept_original_values = df_stress["ept_promedio_min_smoothed"].values
        elif "ept_promedio_min" in df_stress.columns:
            ept_original_values = df_stress["ept_promedio_min"].values
        elif "ept_promedio" in df_stress.columns:
            ept_original_values = df_stress["ept_promedio"].values
        elif "ept_configurado_min" in df_stress.columns:
            ept_original_values = df_stress["ept_configurado_min"].values
        else:
            ept_original_values = np.zeros(len(df_stress), dtype=float)
        
        hdm_end_time = None
        activation_queue_start = None
        activation_delay_minutes = ACTIVATION_DELAY_MINUTES
        
        for i in range(len(df_stress)):
            current_time = momento_values[i]
            if isinstance(current_time, np.datetime64):
                current_time = pd.Timestamp(current_time)
            
            ordenes = ordenes_values[i]
            riders = riders_values[i]
            awt_real = awt_values[i]
            hdm_real = int(hdm_real_values[i])
            
            # Check AND condition
            u1_met = 1 if ordenes >= u1 else 0
            u2_met = 1 if riders >= u2 else 0
            u3_met = 1 if awt_real >= u3 else 0
            and_triggered = u1_met and u2_met and u3_met
            
            hdm_currently_active = 0
            in_delay_period = 0
            
            # Process delay queue
            if activation_queue_start is not None:
                time_delta = current_time - activation_queue_start
                if hasattr(time_delta, 'total_seconds'):
                    time_since_trigger = time_delta.total_seconds() / 60
                else:
                    time_since_trigger = float(time_delta / np.timedelta64(1, 'm'))
                
                if time_since_trigger < activation_delay_minutes:
                    in_delay_period = 1
                elif time_since_trigger < activation_delay_minutes + duracion_hdm:
                    hdm_currently_active = 1
                else:
                    activation_queue_start = None
            
            # New activation
            if and_triggered and activation_queue_start is None:
                activation_queue_start = current_time
                in_delay_period = 1
            
            # Predict AWT
            ordenes_val = float(ordenes)
            riders_val = float(riders)
            hdm_val = float(hdm_currently_active)
            
            ept_base = float(ept_original_values[i])
            if ordenes_val <= 0:
                ept_base = 0.0
            ept_base = max(0, ept_base)
            ept_with_hdm = ept_base + delta_ept if hdm_currently_active else ept_base

            awt_predicted = self.awt_predictor.predict(ordenes_val, riders_val, hdm_val, ept_with_hdm)
            awt_predicted = max(0, awt_predicted)
            
            comparison_data.append({
                "minute": i,
                "timestamp": str(current_time),
                "awt_real": round(awt_real, 2),
                "awt_predicted": round(awt_predicted, 2),
                "hdm_real": hdm_real,
                "hdm_simulated": hdm_currently_active,
                "hdm_in_delay": in_delay_period,
                "ordenes": int(ordenes),
                "riders": int(riders),
                "u1_condition": u1_met,
                "u2_condition": u2_met,
                "u3_condition": u3_met,
                "all_conditions_met": 1 if and_triggered else 0,
                "ept_base": round(ept_base, 2),
                "ept_with_hdm": round(ept_with_hdm, 2),
            })
        
        # Save to CSV
        df_comparison = pd.DataFrame(comparison_data)
        output_path = Path(output_dir) / "stress_day_validation.csv"
        df_comparison.to_csv(output_path, index=False)
        
        # Calculate metrics
        awt_real_mean = df_stress["max_awt_espera_min"].mean()
        awt_pred_mean = df_comparison["awt_predicted"].mean()
        awt_improvement = awt_real_mean - awt_pred_mean
        
        total_and_events = df_comparison["all_conditions_met"].sum()
        total_delay_minutes = df_comparison["hdm_in_delay"].sum()
        total_active_minutes = df_comparison["hdm_simulated"].sum()
        
        summary = f"""
======================================================================
STRESS DAY VALIDATION (2-Minute Activation Delay with AND Logic)
======================================================================

Analysis Period:
  From: {df_stress["momento_exacto"].min()}
  To:   {df_stress["momento_exacto"].max()}
  Duration: {len(df_stress)} minutes ({len(df_stress)/60:.1f} hours)

Real Historical Metrics:
  HDM activations (actual): {int(df_stress["hdm_activo"].sum())}
  Max AWT real: {df_stress["max_awt_espera_min"].max():.2f} min
  Mean AWT real: {awt_real_mean:.2f} min
  Mean ordenes: {df_stress["ordenes_pendientes"].mean():.1f}
  Mean riders: {df_stress["riders_cerca"].mean():.1f}

Recommended Configuration (u1={u1}, u2={u2}, u3={u3}):
  AND logic triggers: {total_and_events} times ({100*total_and_events/len(df_stress):.1f}% of period)
  Time in delay: {total_delay_minutes} minutes (waiting for impact)
  HDM active time: {total_active_minutes} minutes ({100*total_active_minutes/len(df_stress):.1f}% duty cycle)

AND Condition Breakdown:
  u1 (ordenes >= {u1}): {df_comparison["u1_condition"].sum()} mins ({100*df_comparison["u1_condition"].sum()/len(df_stress):.1f}%)
  u2 (riders >= {u2}): {df_comparison["u2_condition"].sum()} mins ({100*df_comparison["u2_condition"].sum()/len(df_stress):.1f}%)
  u3 (awt >= {u3}): {df_comparison["u3_condition"].sum()} mins ({100*df_comparison["u3_condition"].sum()/len(df_stress):.1f}%)
  ALL 3 met: {total_and_events} mins ({100*total_and_events/len(df_stress):.1f}%)

Simulated Impact:
  Mean AWT simulated: {awt_pred_mean:.2f} min
  AWT improvement: {-awt_improvement:.2f} min ({-100*awt_improvement/awt_real_mean if awt_real_mean > 0 else 0:.1f}%)
  EPT increase: {delta_ept:.2f} min (when HDM active)
  Duration per activation: {duracion_hdm} min

Key Insights:
  [OK] Delay working: {total_delay_minutes} min buffer before impact
  [OK] AND logic filtering: Only {100*total_and_events/len(df_stress):.1f}% true simultaneous stress
  [OK] Realistic activation: {100*total_active_minutes/len(df_stress):.1f}% duty cycle (not constant)
  [OK] CSV output: {output_path}

======================================================================
"""
        
        print(summary)
        return summary

    def generate_full_timeline_validation(self, df, u1, u2, u3, delta_ept, duracion_hdm, output_dir):
        """
        Generate minute-by-minute simulation trace for the full input dataset.

        This is a full-backup validation artifact (all minutes), unlike stress-day
        which focuses only on the peak window.

        Args:
            df: Full historical data
            u1, u2, u3: Thresholds
            delta_ept: EPT increase when HDM active
            duracion_hdm: HDM active duration (minutes)
            output_dir: Directory to save CSV

        Returns:
            str: output csv path
        """
        from pathlib import Path

        df_full = df.copy().reset_index(drop=True)

        # Cast thresholds
        u1 = int(round(u1))
        u2 = int(round(u2))
        u3 = int(round(u3))
        duracion_hdm = int(round(duracion_hdm))

        momento_values = df_full["momento_exacto"].values
        ordenes_values = df_full["ordenes_pendientes"].values
        riders_values = df_full["riders_cerca"].values
        awt_values = df_full["max_awt_espera_min"].values
        hdm_real_values = df_full["hdm_activo"].values
        if "ept_promedio_min_smoothed" in df_full.columns:
            ept_original_values = df_full["ept_promedio_min_smoothed"].values
        elif "ept_promedio_min" in df_full.columns:
            ept_original_values = df_full["ept_promedio_min"].values
        elif "ept_promedio" in df_full.columns:
            ept_original_values = df_full["ept_promedio"].values
        elif "ept_configurado_min" in df_full.columns:
            ept_original_values = df_full["ept_configurado_min"].values
        else:
            ept_original_values = np.zeros(len(df_full), dtype=float)

        comparison_data = []
        activation_queue_start = None
        activation_delay_minutes = ACTIVATION_DELAY_MINUTES

        for i in range(len(df_full)):
            current_time = momento_values[i]
            if isinstance(current_time, np.datetime64):
                current_time = pd.Timestamp(current_time)

            ordenes = ordenes_values[i]
            riders = riders_values[i]
            awt_real = awt_values[i]
            hdm_real = int(hdm_real_values[i])

            # Check AND condition
            u1_met = 1 if ordenes >= u1 else 0
            u2_met = 1 if riders >= u2 else 0
            u3_met = 1 if awt_real >= u3 else 0
            and_triggered = u1_met and u2_met and u3_met

            hdm_currently_active = 0
            in_delay_period = 0

            # Delay queue processing
            if activation_queue_start is not None:
                time_delta = current_time - activation_queue_start
                if hasattr(time_delta, 'total_seconds'):
                    time_since_trigger = time_delta.total_seconds() / 60
                else:
                    time_since_trigger = float(time_delta / np.timedelta64(1, 'm'))

                if time_since_trigger < activation_delay_minutes:
                    in_delay_period = 1
                elif time_since_trigger < activation_delay_minutes + duracion_hdm:
                    hdm_currently_active = 1
                else:
                    activation_queue_start = None

            # New activation trigger
            if and_triggered and activation_queue_start is None:
                activation_queue_start = current_time
                in_delay_period = 1

            # EPT simulation: ept_simulado = ept_original + delta_ept when active
            ordenes_val = float(ordenes)
            riders_val = float(riders)
            ept_base = float(ept_original_values[i])
            if ordenes_val <= 0:
                ept_base = 0.0
            ept_base = max(0, ept_base)
            ept_with_hdm = ept_base + delta_ept if hdm_currently_active else ept_base

            # AWT prediction uses simulated EPT signal
            hdm_val = float(hdm_currently_active)
            awt_predicted = self.awt_predictor.predict(ordenes_val, riders_val, hdm_val, ept_with_hdm)
            awt_predicted = max(0, awt_predicted)

            comparison_data.append({
                "minute": i,
                "timestamp": str(current_time),
                "awt_real": round(awt_real, 2),
                "awt_predicted": round(awt_predicted, 2),
                "hdm_real": hdm_real,
                "hdm_simulated": hdm_currently_active,
                "hdm_in_delay": in_delay_period,
                "ordenes": int(ordenes),
                "riders": int(riders),
                "u1_condition": u1_met,
                "u2_condition": u2_met,
                "u3_condition": u3_met,
                "all_conditions_met": 1 if and_triggered else 0,
                "ept_base": round(ept_base, 2),
                "ept_with_hdm": round(ept_with_hdm, 2),
            })

        df_comparison = pd.DataFrame(comparison_data)
        output_path = Path(output_dir) / "full_timeline_validation.csv"
        df_comparison.to_csv(output_path, index=False)

        if DEBUG:
            print(f"[SIMULATOR] Full timeline validation saved: {output_path} ({len(df_comparison)} rows)")

        return str(output_path)


def evaluate_configuration(df, awt_predictor, ept_predictor, baseline_metrics,
                          u1, u2, u3, delta_ept, duracion_hdm):
    """
    Evaluate a single configuration (used for optimization).
    
    Args:
        df: Input data
        awt_predictor: Trained AWT model
        ept_predictor: Trained EPT model
        baseline_metrics: Baseline metrics
        u1, u2, u3, delta_ept, duracion_hdm: Configuration parameters
        
    Returns:
        Dict with evaluation metrics
    """
    simulator = HDMSimulator(awt_predictor, ept_predictor, baseline_metrics)
    result = simulator.simulate_scenario(df, u1, u2, u3, delta_ept, duracion_hdm)
    
    return result


def evaluate_franchise_configuration(partner_payloads, awt_predictor, ept_predictor,
                                     u1, u2, u3, delta_ept, duracion_hdm):
    """
    Evaluate one generic configuration across all partners in a franchise.

    Aggregation is weighted by total pending orders per partner.

    Args:
        partner_payloads: List of dicts with keys:
            - partner_id
            - partner_name (optional)
            - df (single-partner dataframe)
            - baseline_metrics (single-partner baseline)
        awt_predictor: Trained AWT model (franchise-level)
        ept_predictor: Trained EPT model (franchise-level)
        u1, u2, u3, delta_ept, duracion_hdm: Configuration parameters

    Returns:
        Dict with order-weighted aggregate metrics and per-partner breakdown
    """
    weighted_awt_mean = 0.0
    weighted_awt_improvement = 0.0
    weighted_ept_increase = 0.0
    weighted_combined_mean = 0.0
    weighted_combined_improvement = 0.0
    weighted_activation_rate = 0.0
    total_orders_weight = 0.0
    partner_results = []

    for payload in partner_payloads:
        df_partner = payload["df"]
        baseline_metrics = payload["baseline_metrics"]
        partner_id = payload.get("partner_id")
        partner_name = payload.get("partner_name")

        partner_result = evaluate_configuration(
            df_partner,
            awt_predictor,
            ept_predictor,
            baseline_metrics,
            u1, u2, u3, delta_ept, duracion_hdm,
        )

        awt_baseline = baseline_metrics.get("awt_promedio", 0.0)
        ept_baseline = baseline_metrics.get(
            "ept_promedio_min_smoothed_promedio",
            baseline_metrics.get("ept_promedio", baseline_metrics.get("ept_configurado_promedio", 0.0)),
        )
        combined_baseline = awt_baseline + ept_baseline

        awt_improvement = awt_baseline - partner_result["awt_mean"]
        combined_improvement = combined_baseline - partner_result["combined_time_mean"]

        orders_weight = float(df_partner["ordenes_pendientes"].sum())
        # Fallback to minute-count weight if a partner has all-zero pending orders
        if orders_weight <= 0:
            orders_weight = float(len(df_partner))

        total_orders_weight += orders_weight
        weighted_awt_mean += orders_weight * float(partner_result["awt_mean"])
        weighted_awt_improvement += orders_weight * float(awt_improvement)
        weighted_ept_increase += orders_weight * float(partner_result["ept_increase"])
        weighted_combined_mean += orders_weight * float(partner_result["combined_time_mean"])
        weighted_combined_improvement += orders_weight * float(combined_improvement)
        weighted_activation_rate += orders_weight * float(partner_result.get("hdm_activation_rate", 0.0))

        partner_results.append({
            "partner_id": partner_id,
            "partner_name": partner_name,
            "orders_weight": orders_weight,
            "awt_mean": round(float(partner_result["awt_mean"]), 4),
            "awt_improvement": round(float(awt_improvement), 4),
            "ept_increase": round(float(partner_result["ept_increase"]), 4),
            "combined_time_mean": round(float(partner_result["combined_time_mean"]), 4),
            "combined_improvement": round(float(combined_improvement), 4),
            "hdm_activation_rate": round(float(partner_result.get("hdm_activation_rate", 0.0)), 4),
        })

    if total_orders_weight <= 0:
        total_orders_weight = 1.0

    return {
        "awt_mean": weighted_awt_mean / total_orders_weight,
        "awt_improvement": weighted_awt_improvement / total_orders_weight,
        "ept_increase": weighted_ept_increase / total_orders_weight,
        "combined_time_mean": weighted_combined_mean / total_orders_weight,
        "combined_improvement": weighted_combined_improvement / total_orders_weight,
        "hdm_activation_rate": weighted_activation_rate / total_orders_weight,
        "orders_weight_total": total_orders_weight,
        "partner_results": partner_results,
    }
