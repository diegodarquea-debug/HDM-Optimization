"""
Main pipeline: Load → Analyze → Train → Simulate → Optimize
Supports both PARTNER-level (single partner) and FRANCHISE-level (all partners) modes.
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from src.config import (
    N_SIMULATIONS,
    N_OPTIMIZATION_CALLS,
    THRESHOLDS,
    OUTPUT_DIR,
    OBJECTIVE_WEIGHTS,
    RANDOM_SEED,
    MAX_EPT_INCREASE,
    BAYESIAN_SETTINGS,
)
from src.data_loader import load_and_prepare_data, get_unique_partners, get_date_range
from src.analytics import analyze_data, calculate_baseline_metrics
from src.model import train_models
from src.simulator import HDMSimulator, evaluate_franchise_configuration
from src.optimizer import optimize_hdm_thresholds

logger = logging.getLogger("hdm_pipeline")

def process_partner(df_partner: pd.DataFrame, partner_id: Any, partner_name: str = None) -> Dict[str, Any]:
    """Process a single partner: analyze, train, simulate, optimize."""
    start_date, end_date = get_date_range(df_partner)
    logger.info(f"Processing PARTNER {partner_id} ({partner_name or ''}) | {start_date.date()} to {end_date.date()}")
    
    analysis = analyze_data(df_partner)
    baseline_metrics = analysis["baseline_metrics"]
    awt_predictor, ept_predictor = train_models(df_partner)
    
    simulator = HDMSimulator(awt_predictor, ept_predictor, baseline_metrics)
    sim_results = simulator.run_simulations(df_partner, THRESHOLDS, n_sims=N_SIMULATIONS)
    
    optimizer, opt_result = optimize_hdm_thresholds(
        df_partner, awt_predictor, ept_predictor, baseline_metrics,
        n_calls=N_OPTIMIZATION_CALLS, method="gp_minimize"
    )
    
    top_3 = optimizer.get_top_3_strategies()
    best_config = top_3.get("Equilibrada") or top_3.get("Agresiva") or (list(top_3.values())[0] if top_3 else {})
    
    return {
        "partner_id": partner_id,
        "partner_name": partner_name,
        "data": df_partner,
        "analysis": analysis,
        "baseline_metrics": baseline_metrics,
        "predictors": (awt_predictor, ept_predictor),
        "simulator": simulator,
        "simulations": sim_results,
        "optimizer": optimizer,
        "optimization_result": opt_result,
        "top_3_strategies": top_3,
        "best_config": best_config,
        "start_date": start_date,
        "end_date": end_date,
    }

def run_franchise_mode(df: pd.DataFrame, all_partners: np.ndarray):
    """Execution flow for franchise mode."""
    logger.info(f"Processing FRANCHISE ({len(all_partners)} partners)")

    analysis = analyze_data(df)
    baseline_metrics = analysis["baseline_metrics"]
    awt_predictor, ept_predictor = train_models(df)

    partner_payloads = []
    for partner_id in all_partners:
        df_p = df[df["partner_id"] == partner_id].copy()
        partner_payloads.append({
            "partner_id": partner_id,
            "partner_name": df_p["partner_name"].iloc[0] if "partner_name" in df_p.columns else None,
            "df": df_p,
            "baseline_metrics": calculate_baseline_metrics(df_p),
        })

    logger.info("Starting Franchise Monte Carlo exploration...")
    simulator = HDMSimulator(awt_predictor, ept_predictor, baseline_metrics)
    mc_df = simulator.run_simulations(df, THRESHOLDS, n_sims=N_SIMULATIONS)

    # Calculate objective score using weights from config
    mc_df["objective_score"] = (OBJECTIVE_WEIGHTS["awt"] * mc_df["awt_improvement"]) - \
                              (OBJECTIVE_WEIGHTS["ept_penalty"] * mc_df["ept_increase"])

    mc_output = OUTPUT_DIR / "monte_carlo_franchise_exploration.csv"
    mc_df.to_csv(mc_output, index=False)

    top_k = BAYESIAN_SETTINGS.get("mc_seed_top_k", 20)
    mc_valid = mc_df[mc_df["ept_increase"] <= MAX_EPT_INCREASE].copy()
    if mc_valid.empty: mc_valid = mc_df.copy()
    mc_top = mc_valid.sort_values("objective_score", ascending=False).head(top_k)
    bayes_x0 = mc_top[["u1", "u2", "u3", "delta_ept", "duracion_hdm"]].values.tolist()

    optimizer, opt_result = optimize_hdm_thresholds(
        df, awt_predictor, ept_predictor, baseline_metrics,
        n_calls=N_OPTIMIZATION_CALLS, method="gp_minimize",
        franchise_payloads=partner_payloads, x0=bayes_x0
    )

    top_3 = optimizer.get_top_3_strategies()
    best_config = top_3.get("Equilibrada") or top_3.get("Agresiva") or (list(top_3.values())[0] if top_3 else {})

    franchise_eval = evaluate_franchise_configuration(
        partner_payloads, awt_predictor, ept_predictor,
        best_config["u1"], best_config["u2"], best_config["u3"],
        best_config["delta_ept"], best_config["duracion_hdm"]
    )

    _save_franchise_outputs(optimizer, best_config, franchise_eval, len(all_partners), OUTPUT_DIR)
    return {
        "analysis": analysis,
        "predictors": (awt_predictor, ept_predictor),
        "optimizer": optimizer,
        "best_config": best_config,
        "franchise_evaluation": franchise_eval,
    }

def main():
    parser = argparse.ArgumentParser(description="HDM Optimization Pipeline")
    parser.add_argument("--mode", choices=["partner", "franchise"], default="franchise")
    parser.add_argument("--partner-id", type=int, default=None)
    args = parser.parse_args()
    
    logger.info(f"HDM OPTIMIZATION PIPELINE - {args.mode.upper()} MODE")
    
    df = load_and_prepare_data(mode=args.mode)
    all_partners = get_unique_partners(df)
    
    if args.mode == "partner":
        partner_id = args.partner_id or all_partners[0]
        df_p = df[df["partner_id"] == partner_id].copy()
        result = process_partner(df_p, partner_id, df_p["partner_name"].iloc[0] if "partner_name" in df_p.columns else None)
        _save_partner_outputs(result, OUTPUT_DIR)
    else:
        run_franchise_mode(df, all_partners)

def _save_partner_outputs(result, output_dir):
    """Save single-partner outputs (legacy format)."""
    optimizer = result["optimizer"]
    best_config = result["best_config"]
    
    pd.DataFrame(optimizer.optimization_history).to_csv(output_dir / "optimization_history.csv", index=False)
    
    rec = {
        "partner_id": result["partner_id"],
        "u1": best_config.get("u1"),
        "u2": best_config.get("u2"),
        "u3": best_config.get("u3"),
        "delta_ept": best_config.get("delta_ept"),
        "duracion_hdm": best_config.get("duracion_hdm"),
        "awt_reduction": best_config.get("awt_improvement"),
        "ept_increase": best_config.get("ept_increase"),
    }
    pd.DataFrame([rec]).to_csv(output_dir / "hdm_recommendations.csv", index=False)
    
    try:
        result["simulator"].generate_stress_day_analysis(result["data"], best_config["u1"], best_config["u2"], best_config["u3"], best_config["delta_ept"], best_config["duracion_hdm"], output_dir)
    except Exception as e:
        logger.warning(f"Validation failed: {e}")

def _save_franchise_outputs(optimizer, best_config, franchise_eval, num_partners, output_dir):
    """Save franchise-level outputs."""
    pd.DataFrame(optimizer.optimization_history).to_csv(output_dir / "optimization_history.csv", index=False)

    best_config["scope"] = "franchise"
    best_config["num_partners"] = num_partners
    pd.DataFrame([best_config]).to_csv(output_dir / "franchise_optimal_config.csv", index=False)

    pd.DataFrame(franchise_eval["partner_results"]).to_csv(output_dir / "franchise_impact_by_partner.csv", index=False)
    logger.info(f"Franchise optimization complete. Partners: {num_partners}")

if __name__ == "__main__":
    main()
