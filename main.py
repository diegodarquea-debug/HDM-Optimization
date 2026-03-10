"""
Main pipeline: Load → Analyze → Train → Simulate → Optimize
Supports both PARTNER-level (single partner) and FRANCHISE-level (all partners) modes.
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.config import (
    DEBUG,
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
from src.simulator import HDMSimulator
from src.simulator import evaluate_franchise_configuration
from src.optimizer import optimize_hdm_thresholds


def process_partner(df_partner, partner_id, partner_name=None):
    """
    Process a single partner: analyze, train, simulate, optimize.
    
    Args:
        df_partner: DataFrame for single partner
        partner_id: Partner ID
        partner_name: Partner name (optional, for reporting)
        
    Returns:
        Dict with results for this partner
    """
    start_date, end_date = get_date_range(df_partner)
    
    if DEBUG:
        print(f"\n  [PARTNER {partner_id}] {partner_name or ''} | "
              f"{start_date.date()} to {end_date.date()} | "
              f"{len(df_partner)} rows")
    
    # Analyze this partner's baseline
    analysis = analyze_data(df_partner)
    baseline_metrics = analysis["baseline_metrics"]
    
    # Train models on this partner's data
    awt_predictor, ept_predictor = train_models(df_partner)
    
    # Simulate configurations for this partner
    simulator = HDMSimulator(awt_predictor, ept_predictor, baseline_metrics)
    sim_results = simulator.run_simulations(df_partner, THRESHOLDS, n_sims=N_SIMULATIONS)
    
    # Optimize thresholds for this partner
    optimizer, opt_result = optimize_hdm_thresholds(
        df_partner, awt_predictor, ept_predictor, baseline_metrics,
        n_calls=N_OPTIMIZATION_CALLS, method="gp_minimize"
    )
    
    # Get best configuration
    top_3_strategies = optimizer.get_top_3_strategies()
    best_config = (top_3_strategies.get("Equilibrada") or 
                   top_3_strategies.get("Agresiva") or 
                   (list(top_3_strategies.values())[0] if top_3_strategies else {}))
    
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
        "top_3_strategies": top_3_strategies,
        "best_config": best_config,
        "start_date": start_date,
        "end_date": end_date,
    }


def main():
    """
    Main execution pipeline (supports partner and franchise modes).
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HDM Optimization Pipeline")
    parser.add_argument("--mode", choices=["partner", "franchise"], default="franchise",
                       help="Execution mode: 'partner' (single) or 'franchise' (all)")
    parser.add_argument("--partner-id", type=int, default=None,
                       help="Specific partner_id to process (partner mode only)")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"HDM OPTIMIZATION PIPELINE - {args.mode.upper()} MODE")
    print("="*70)
    
    # ===== STEP 1: LOAD DATA =====
    print("\n[STEP 1] Loading and preparing data...")
    df = load_and_prepare_data(mode=args.mode)
    all_partners = get_unique_partners(df)
    all_start_date, all_end_date = get_date_range(df)
    
    print(f"\nDataset Summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique partners: {len(all_partners)}")
    print(f"  Date range: {all_start_date.date()} to {all_end_date.date()}")
    print(f"  Date span: {(all_end_date - all_start_date).total_seconds() / 3600:.1f} hours")
    
    # ===== SELECT PROCESSING MODE =====
    if args.mode == "partner":
        # PARTNER MODE: Process single partner
        partner_id = args.partner_id or all_partners[0]
        if partner_id not in all_partners:
            print(f"ERROR: Partner {partner_id} not found in dataset.")
            sys.exit(1)
        
        df_partner = df[df["partner_id"] == partner_id].copy()
        partner_name = df_partner["partner_name"].iloc[0] if "partner_name" in df_partner.columns else None
        
        print(f"\n[STEP 2] Processing PARTNER {partner_id} ({partner_name})...")
        result = process_partner(df_partner, partner_id, partner_name)
        
        # Save single-partner outputs (legacy flow)
        _save_partner_outputs(result, OUTPUT_DIR)
        
    else:
        # FRANCHISE MODE: Single generic configuration optimized across all partners
        print(f"\n[STEP 2] Processing FRANCHISE ({len(all_partners)} partners)...")
        print(f"  Partners to process: {all_partners}")

        # Global analysis/model for chain behavior
        print("\n[STEP 3] Running franchise-level analysis...")
        analysis = analyze_data(df)
        baseline_metrics = analysis["baseline_metrics"]

        print("\n[STEP 4] Training franchise-level predictive models...")
        awt_predictor, ept_predictor = train_models(df)

        # Build per-partner payloads for weighted objective evaluation
        partner_payloads = []
        for idx, partner_id in enumerate(all_partners, 1):
            print(f"  [{idx}/{len(all_partners)}] Preparing partner {partner_id} payload...")
            df_partner = df[df["partner_id"] == partner_id].copy()
            partner_name = df_partner["partner_name"].iloc[0] if "partner_name" in df_partner.columns else None
            partner_payloads.append({
                "partner_id": partner_id,
                "partner_name": partner_name,
                "df": df_partner,
                "baseline_metrics": calculate_baseline_metrics(df_partner),
            })

        print("\n[STEP 5] Optimizing one generic configuration for full franchise...")

        # Monte Carlo exploration before Bayesian optimization (franchise-wide)
        print(f"\n[STEP 5A] Franchise Monte Carlo exploration ({N_SIMULATIONS} configs)...")
        mc_df = _run_franchise_monte_carlo(
            partner_payloads,
            awt_predictor,
            ept_predictor,
            n_sims=N_SIMULATIONS,
            random_seed=RANDOM_SEED,
        )
        mc_output = OUTPUT_DIR / "monte_carlo_franchise_exploration.csv"
        mc_df.to_csv(mc_output, index=False)
        print(f"  Monte Carlo exploration saved to: {mc_output}")

        # Use top Monte Carlo configurations as seeds for Bayes
        mc_valid = mc_df[mc_df["ept_increase"] <= MAX_EPT_INCREASE].copy()
        if mc_valid.empty:
            mc_valid = mc_df.copy()
        top_k = int(BAYESIAN_SETTINGS.get("mc_seed_top_k", 15))
        mc_top = mc_valid.sort_values("objective_score", ascending=False).head(min(top_k, len(mc_valid)))
        bayes_x0 = mc_top[["u1", "u2", "u3", "delta_ept", "duracion_hdm"]].values.tolist()

        print(f"\n[STEP 5B] Bayesian optimization seeded with {len(bayes_x0)} Monte Carlo configs...")
        optimizer, opt_result = optimize_hdm_thresholds(
            df,
            awt_predictor,
            ept_predictor,
            baseline_metrics,
            n_calls=N_OPTIMIZATION_CALLS,
            method="gp_minimize",
            franchise_payloads=partner_payloads,
            x0=bayes_x0,
        )

        # Select best recommended strategic profile
        top_3_strategies = optimizer.get_top_3_strategies()
        best_config = (top_3_strategies.get("Equilibrada") or
                       top_3_strategies.get("Agresiva") or
                       (list(top_3_strategies.values())[0] if top_3_strategies else {}))

        # Evaluate final selected config across all partners
        franchise_eval = evaluate_franchise_configuration(
            partner_payloads,
            awt_predictor,
            ept_predictor,
            best_config.get("u1", 0),
            best_config.get("u2", 0),
            best_config.get("u3", 0),
            best_config.get("delta_ept", 0),
            best_config.get("duracion_hdm", 0),
        )

        _save_franchise_outputs(
            optimizer,
            best_config,
            franchise_eval,
            len(all_partners),
            OUTPUT_DIR,
        )

        result = {
            "analysis": analysis,
            "predictors": (awt_predictor, ept_predictor),
            "optimizer": optimizer,
            "optimization_result": opt_result,
            "best_config": best_config,
            "franchise_evaluation": franchise_eval,
        }
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70 + "\n")
    
    return result


def _save_partner_outputs(result, output_dir):
    
    """Save single-partner outputs (legacy format)."""
    result = result  # Reference single partner result
    optimizer = result["optimizer"]
    best_config = result["best_config"]
    baseline_metrics = result["baseline_metrics"]
    df_partner = result["data"]
    start_date = result["start_date"]
    end_date = result["end_date"]
    simulator = result["simulator"]
    
    # Save optimization history
    opt_history_df = pd.DataFrame(optimizer.optimization_history)
    numeric_cols = ["delta_ept", "awt_mean", "awt_improvement", "ept_increase", 
                    "combined_improvement", "hdm_activation_rate", "total_loss"]
    for col in numeric_cols:
        if col in opt_history_df.columns:
            opt_history_df[col] = opt_history_df[col].round(2)
    opt_output = output_dir / "optimization_history.csv"
    opt_history_df.to_csv(opt_output, index=False)
    
    # Save strategic recommendations
    top_3_strategies = result["top_3_strategies"]
    strategies_list = []
    for strategy_name, config in top_3_strategies.items():
        rec = {
            "perfil_estrategico": strategy_name,
            "u1": config["u1"],
            "u2": config["u2"],
            "u3": config["u3"],
            "delta_ept": round(config["delta_ept"], 2),
            "duracion_hdm": config["duracion_hdm"],
            "awt_improvement": round(config["awt_improvement"], 2),
            "awt_improvement_pct": round(
                config["awt_improvement"] / baseline_metrics["awt_promedio"] * 100, 2
            ) if baseline_metrics["awt_promedio"] > 0 else 0,
            "ept_increase": round(config["ept_increase"], 2),
            "combined_improvement": round(config["combined_improvement"], 2),
            "hdm_activation_rate": round(config["hdm_activation_rate"], 2),
        }
        strategies_list.append(rec)
    
    strategies_df = pd.DataFrame(strategies_list)
    strategies_output = output_dir / "hdm_strategic_recommendations.csv"
    strategies_df.to_csv(strategies_output, index=False)
    
    # Single best recommendation
    recommendation = {
        "partner_id": result["partner_id"],
        "date_range_start": str(start_date),
        "date_range_end": str(end_date),
        "u1_ordenes_threshold": best_config.get("u1", 0),
        "u2_riders_threshold": best_config.get("u2", 0),
        "u3_awt_threshold_min": best_config.get("u3", 0),
        "delta_ept_min": round(best_config.get("delta_ept", 0), 2),
        "duracion_hdm_min": best_config.get("duracion_hdm", 0),
        "expected_awt_reduction": round(best_config.get("awt_improvement", 0), 2),
        "expected_awt_reduction_pct": round(
            best_config.get("awt_improvement", 0) / baseline_metrics["awt_promedio"] * 100, 2
        ) if baseline_metrics["awt_promedio"] > 0 else 0,
        "ept_increase": round(best_config.get("ept_increase", 0), 2),
        "expected_hdm_activation_rate": round(best_config.get("hdm_activation_rate", 0), 2),
        "combined_time_reduction": round(best_config.get("combined_improvement", 0), 2),
    }
    
    rec_df = pd.DataFrame([recommendation])
    rec_output = output_dir / "hdm_recommendations.csv"
    rec_df.to_csv(rec_output, index=False)
    
    print(f"\n  Outputs saved to {output_dir}")
    
    # Stress day validation
    try:
        simulator.generate_stress_day_analysis(
            df_partner,
            u1=best_config.get("u1", 0),
            u2=best_config.get("u2", 0),
            u3=best_config.get("u3", 0),
            delta_ept=best_config.get("delta_ept", 0),
            duracion_hdm=best_config.get("duracion_hdm", 0),
            output_dir=output_dir
        )
        simulator.generate_full_timeline_validation(
            df_partner,
            u1=best_config.get("u1", 0),
            u2=best_config.get("u2", 0),
            u3=best_config.get("u3", 0),
            delta_ept=best_config.get("delta_ept", 0),
            duracion_hdm=best_config.get("duracion_hdm", 0),
            output_dir=output_dir
        )
    except Exception as e:
        print(f"  WARNING: Validation generation failed: {e}")


def _run_franchise_monte_carlo(partner_payloads, awt_predictor, ept_predictor, n_sims, random_seed):
    """Random exploration of franchise-wide configurations before Bayes refinement."""
    rng = np.random.default_rng(random_seed)
    rows = []

    # Progress bar for Monte Carlo
    pbar = tqdm(total=n_sims, desc="[SIMULATOR] Monte Carlo Exploration", unit="config")
    
    for _ in range(n_sims):
        u1 = int(rng.integers(int(THRESHOLDS["u1"][0]), int(THRESHOLDS["u1"][1]) + 1))
        u2 = int(rng.integers(int(THRESHOLDS["u2"][0]), int(THRESHOLDS["u2"][1]) + 1))
        u3 = int(rng.integers(int(THRESHOLDS["u3"][0]), int(THRESHOLDS["u3"][1]) + 1))
        delta_ept = float(rng.choice(THRESHOLDS["delta_ept"]))
        duracion_hdm = int(rng.integers(int(THRESHOLDS["duracion_hdm"][0]), int(THRESHOLDS["duracion_hdm"][1]) + 1))

        eval_result = evaluate_franchise_configuration(
            partner_payloads,
            awt_predictor,
            ept_predictor,
            u1, u2, u3, delta_ept, duracion_hdm,
        )

        objective_score = (
            OBJECTIVE_WEIGHTS.get("awt", 1.0) * float(eval_result.get("awt_improvement", 0.0))
            - OBJECTIVE_WEIGHTS.get("ept_penalty", 0.4) * float(eval_result.get("ept_increase", 0.0))
        )

        rows.append({
            "u1": u1,
            "u2": u2,
            "u3": u3,
            "delta_ept": delta_ept,
            "duracion_hdm": duracion_hdm,
            "awt_mean": float(eval_result.get("awt_mean", 0.0)),
            "awt_improvement": float(eval_result.get("awt_improvement", 0.0)),
            "ept_increase": float(eval_result.get("ept_increase", 0.0)),
            "combined_time_mean": float(eval_result.get("combined_time_mean", 0.0)),
            "combined_improvement": float(eval_result.get("combined_improvement", 0.0)),
            "hdm_activation_rate": float(eval_result.get("hdm_activation_rate", 0.0)),
            "orders_weight_total": float(eval_result.get("orders_weight_total", 0.0)),
            "objective_score": float(objective_score),
        })
        pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


def _save_franchise_outputs(optimizer, best_config, franchise_eval, num_partners, output_dir):
    """Save franchise-level outputs for one generic configuration."""
    # Save optimization history
    opt_history_df = pd.DataFrame(optimizer.optimization_history)
    numeric_cols = [
        "delta_ept", "awt_mean", "awt_improvement", "ept_increase",
        "combined_improvement", "hdm_activation_rate", "orders_weight_total", "total_loss"
    ]
    for col in numeric_cols:
        if col in opt_history_df.columns:
            opt_history_df[col] = opt_history_df[col].round(4)
    opt_output = output_dir / "optimization_history.csv"
    opt_history_df.to_csv(opt_output, index=False)

    # Save final one-config recommendation for franchise
    optimal_config_df = pd.DataFrame([{
        "scope": "franchise",
        "u1": int(best_config.get("u1", 0)),
        "u2": int(best_config.get("u2", 0)),
        "u3": int(best_config.get("u3", 0)),
        "delta_ept": round(float(best_config.get("delta_ept", 0)), 2),
        "duracion_hdm": int(best_config.get("duracion_hdm", 0)),
        "awt_improvement_weighted": round(float(franchise_eval.get("awt_improvement", 0)), 4),
        "ept_increase_weighted": round(float(franchise_eval.get("ept_increase", 0)), 4),
        "combined_improvement_weighted": round(float(franchise_eval.get("combined_improvement", 0)), 4),
        "hdm_activation_rate_weighted": round(float(franchise_eval.get("hdm_activation_rate", 0)), 4),
        "orders_weight_total": int(franchise_eval.get("orders_weight_total", 0)),
        "num_partners": int(num_partners),
    }])
    optimal_output = output_dir / "franchise_optimal_config.csv"
    optimal_config_df.to_csv(optimal_output, index=False)

    # Save partner-level impact using the same generic franchise config
    impact_df = pd.DataFrame(franchise_eval.get("partner_results", []))
    impact_output = output_dir / "franchise_impact_by_partner.csv"
    impact_df.to_csv(impact_output, index=False)

    # Backward-compatible aggregate summary
    aggregate_summary = pd.DataFrame([{
        "metric": "franchise_objective_score",
        "value": round(float(franchise_eval.get("combined_improvement", 0)), 4),
        "total_orders": int(franchise_eval.get("orders_weight_total", 0)),
        "num_partners": int(num_partners),
    }])
    aggregate_output = output_dir / "aggregate_objective_summary.csv"
    aggregate_summary.to_csv(aggregate_output, index=False)

    print(f"\n  Franchise Summary:")
    print(f"    • Partners processed: {num_partners}")
    print(f"    • Total orders weight: {int(franchise_eval.get('orders_weight_total', 0))}")
    print(f"    • Weighted combined improvement: {float(franchise_eval.get('combined_improvement', 0)):.4f}")
    print(f"    • Saved to: {opt_output}")
    print(f"    • Saved to: {optimal_output}")
    print(f"    • Saved to: {impact_output}")
    print(f"    • Saved to: {aggregate_output}")


if __name__ == "__main__":
    results = main()
