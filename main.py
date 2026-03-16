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
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm

from src.config import (
    N_SIMULATIONS,
    N_OPTIMIZATION_CALLS,
    THRESHOLDS,
    OUTPUT_DIR,
    OBJECTIVE_WEIGHTS,
    RANDOM_SEED,
    MAX_EPT_INCREASE,
    BAYESIAN_SETTINGS,
    DATA_SOURCE,
    configure_logging,
)
from src.data_loader import load_and_prepare_data, get_unique_partners, get_date_range
from src.analytics import analyze_data, calculate_baseline_metrics
from src.model import train_models
from src.simulator import HDMSimulator, evaluate_franchise_configuration
from src.optimizer import optimize_hdm_thresholds

logger = logging.getLogger("hdm_pipeline")


def _get_versioned_output_dir(output_base_dir: Path = OUTPUT_DIR) -> Path:
    """
    Create versioned output directory for this run.
    Format: outputs/runs/YYYYMMDD_HHMMSS_run/
    Ensures incremental persistence for JupyterHub sessions that may be interrupted.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = output_base_dir / "runs" / f"{timestamp}_run"
    versioned_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Versioned output directory: {versioned_dir}")
    return versioned_dir


def _prompt_if_missing(current_value: Any, prompt_text: str, default: Any = None) -> Any:
    if current_value is not None and str(current_value).strip() != "":
        return current_value

    if not sys.stdin.isatty():
        if default is not None:
            return default
        raise ValueError(f"Missing required input for: {prompt_text}")

    suffix = f" [{default}]" if default is not None else ""
    raw_value = input(f"{prompt_text}{suffix}: ").strip()
    if raw_value:
        return raw_value
    return default


def _collect_bigquery_query_inputs(args, bq_query: str):
    """
    Collect BigQuery query parameters from args or prompts.
    For franchise/grade/dates: prompt if not provided.
    For country_id/entity_id: use provided values (typically defaults from args).
    """
    requires_real_query_inputs = bool(bq_query) and any(
        placeholder in bq_query
        for placeholder in [
            "@target_franchise",
            "@target_grade",
            "@start_date",
            "@end_date",
        ]
    )

    if not requires_real_query_inputs:
        return args, {}

    # These three MUST be provided by user
    args.franchise = _prompt_if_missing(args.franchise, "Franchise", default="KFC")
    args.grade = _prompt_if_missing(args.grade, "Grade", default="AAA")
    args.start_date = _prompt_if_missing(args.start_date, "Start date (YYYY-MM-DD)")
    args.end_date = _prompt_if_missing(args.end_date, "End date (YYYY-MM-DD)")

    # These are fixed (from args defaults), no prompting needed
    # But we include them in query_parameters for BigQuery
    query_parameters = {
        "target_franchise": str(args.franchise),
        "target_grade": str(args.grade),
        "target_country_id": int(args.country_id),  # Always has default=2
        "target_entity_id": str(args.entity_id),    # Always has default='PY_CL'
    }
    return args, query_parameters

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

def run_franchise_mode(
    df: pd.DataFrame,
    all_partners: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
    optimization_scope: str = "global",
):
    """Execution flow for franchise mode with global or clustered optimization."""
    logger.info(
        f"Processing FRANCHISE ({len(all_partners)} partners) with {optimization_scope.upper()} optimization"
    )

    if df.empty or len(all_partners) == 0:
        raise ValueError(
            "No rows were returned for franchise mode. Check BigQuery filters (franchise/grade/entity_id/date range/country_id)."
        )

    cluster_specs = []
    if optimization_scope == "global":
        cluster_specs = [("Global", all_partners, df.copy())]
    else:
        partner_stats = []
        for pid in all_partners:
            df_p = df[df["partner_id"] == pid]
            partner_stats.append({
                "partner_id": pid,
                "avg_orders": df_p["ordenes_pendientes"].mean(),
                "avg_awt": df_p["max_awt_espera_min"].mean(),
                "count": len(df_p)
            })
        df_stats = pd.DataFrame(partner_stats)

        if df_stats.empty:
            raise ValueError("No partner statistics available to build clusters. Input dataset is empty after filters.")

        median_orders = df_stats["avg_orders"].median()
        df_stats["cluster"] = np.where(df_stats["avg_orders"] >= median_orders, "HighVolume", "LowVolume")

        clusters = df_stats["cluster"].unique()
        cluster_list = list(clusters) + ["Global"]
        for cluster_name in cluster_list:
            if cluster_name == "Global":
                cluster_partners = all_partners
                df_cluster = df.copy()
            else:
                cluster_partners = df_stats[df_stats["cluster"] == cluster_name]["partner_id"].values
                df_cluster = df[df["partner_id"].isin(cluster_partners)].copy()
            cluster_specs.append((cluster_name, cluster_partners, df_cluster))

    simulation_budget = N_SIMULATIONS if optimization_scope == "global" else max(1, N_SIMULATIONS // 2)
    optimization_budget = N_OPTIMIZATION_CALLS if optimization_scope == "global" else max(1, N_OPTIMIZATION_CALLS // 2)

    all_cluster_results = []
    total_progress_steps = max(1, len(cluster_specs) * (simulation_budget + optimization_budget))
    progress_bar = tqdm(total=total_progress_steps, desc="Franchise optimization", dynamic_ncols=True)

    try:
        for cluster_name, cluster_partners, df_cluster in cluster_specs:

            logger.info(f"Optimizing Cluster: {cluster_name} ({len(cluster_partners)} partners)")

            analysis = analyze_data(df_cluster)
            baseline_metrics = analysis["baseline_metrics"]
            awt_predictor, ept_predictor = train_models(df_cluster)

            partner_payloads = []
            for pid in cluster_partners:
                df_p = df[df["partner_id"] == pid].copy()
                partner_payloads.append({
                    "partner_id": pid,
                    "partner_name": df_p["partner_name"].iloc[0] if "partner_name" in df_p.columns else None,
                    "df": df_p,
                    "baseline_metrics": calculate_baseline_metrics(df_p),
                })

            def simulation_progress(_completed, _total, current_cluster=cluster_name):
                progress_bar.set_description(f"{current_cluster}: simulating")
                progress_bar.update(1)

            simulator = HDMSimulator(awt_predictor, ept_predictor, baseline_metrics)
            mc_df = simulator.run_simulations(
                df_cluster,
                THRESHOLDS,
                n_sims=simulation_budget,
                progress_callback=simulation_progress,
            )
            mc_df["objective_score"] = (OBJECTIVE_WEIGHTS["awt"] * mc_df["awt_improvement"]) - (OBJECTIVE_WEIGHTS["ept_penalty"] * mc_df["ept_increase"])

            top_k = BAYESIAN_SETTINGS.get("mc_seed_top_k", 20)
            mc_valid = mc_df[mc_df["ept_increase"] <= MAX_EPT_INCREASE].copy()
            if mc_valid.empty: mc_valid = mc_df.copy()
            mc_top = mc_valid.sort_values("objective_score", ascending=False).head(top_k)
            bayes_x0 = mc_top[["u1", "u2", "u3", "delta_ept", "duracion_hdm"]].values.tolist()

            def optimization_progress(_completed, _total, current_cluster=cluster_name):
                progress_bar.set_description(f"{current_cluster}: optimizing")
                progress_bar.update(1)

            optimizer, opt_result = optimize_hdm_thresholds(
                df_cluster, awt_predictor, ept_predictor, baseline_metrics,
                n_calls=optimization_budget, method="gp_minimize",
                franchise_payloads=partner_payloads, x0=bayes_x0,
                progress_callback=optimization_progress,
            )

            top_3 = optimizer.get_top_3_strategies()
            best_config = top_3.get("Equilibrada") or top_3.get("Agresiva") or (list(top_3.values())[0] if top_3 else {})
            best_config["cluster"] = cluster_name

            franchise_eval = evaluate_franchise_configuration(
                partner_payloads, awt_predictor, ept_predictor,
                best_config["u1"], best_config["u2"], best_config["u3"],
                best_config["delta_ept"], best_config["duracion_hdm"]
            )

            all_cluster_results.append({
                "cluster": cluster_name,
                "best_config": best_config,
                "evaluation": franchise_eval,
                "optimizer": optimizer
            })
    finally:
        progress_bar.close()

    _save_clustered_outputs(all_cluster_results, output_dir)
    return all_cluster_results

def main():
    configure_logging()

    parser = argparse.ArgumentParser(
        description="HDM Optimization Pipeline - Franchise Mode (Chile/KFC)"
    )
    
    # **REQUIRED PARAMETERS** (user must specify these)
    parser.add_argument("--franchise", type=str, required=True,
                        help="Franchise name (e.g. KFC, Burger King, etc.)")
    parser.add_argument("--grade", type=str, required=True,
                        help="Vendor grade (AAA, AA, A)")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    
    # FIXED PARAMETERS (for Chile/KFC backtesting - change only if needed)
    parser.add_argument("--mode", choices=["partner", "franchise"], default="franchise", 
                        help="Execution mode (default: franchise)")
    parser.add_argument("--data-source", choices=["auto", "csv", "bigquery"], default="bigquery",
                        help="Data source (default: bigquery)")
    parser.add_argument("--bq-query-file", type=str, default="sql/franchise_input.sql",
                        help="BigQuery SQL file (default: sql/franchise_input.sql)")
    parser.add_argument("--country-id", type=int, default=2,
                        help="Country ID (default: 2 for Chile)")
    parser.add_argument("--entity-id", type=str, default="PY_CL",
                        help="Entity ID (default: PY_CL for Chile)")
    parser.add_argument("--optimization-scope", choices=["global", "clustered"], default="global",
                        help="Optimization scope for franchise mode (default: global)")
    
    # OPTIONAL PARAMETERS (for advanced usage)
    parser.add_argument("--partner-id", type=int, default=None,
                        help="Single partner ID (for partner mode)")
    parser.add_argument("--partner-ids", type=str, default=None,
                        help="Comma-separated partner IDs for franchise subset")
    parser.add_argument("--data-file", type=str, default=None,
                        help="CSV input path (used if data-source=csv)")
    args = parser.parse_args()

    logger.info(f"HDM OPTIMIZATION PIPELINE - {args.mode.upper()} MODE")
    logger.info(
        f"Execution settings: N_SIMULATIONS={N_SIMULATIONS}, "
        f"N_OPTIMIZATION_CALLS={N_OPTIMIZATION_CALLS}, "
        f"OPTIMIZATION_SCOPE={args.optimization_scope}"
    )

    partner_ids_filter = [int(pid.strip()) for pid in args.partner_ids.split(",")] if args.partner_ids else []

    bq_query = None
    if args.bq_query_file:
        bq_query_path = Path(args.bq_query_file)
        if not bq_query_path.exists():
            raise FileNotFoundError(f"BigQuery query file not found: {bq_query_path}")
        bq_query = bq_query_path.read_text(encoding="utf-8")

    args, query_parameters = _collect_bigquery_query_inputs(args, bq_query)

    # Convert dates AFTER collecting inputs (in case prompts updated them)
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None

    df = load_and_prepare_data(
        filepath=args.data_file,
        partner_id=args.partner_id,
        start_date=pd.to_datetime(args.start_date) if args.start_date else None,
        end_date=pd.to_datetime(args.end_date) if args.end_date else None,
        mode=args.mode,
        source=args.data_source,
        bigquery_query=bq_query,
        partner_ids=partner_ids_filter,
        query_parameters=query_parameters,
    )

    if args.mode == "franchise" and partner_ids_filter:
        df = df[df["partner_id"].isin(partner_ids_filter)].copy()
        if df.empty:
            raise ValueError("No data left after applying --partner-ids filter.")

    if df.empty:
        details = [
            f"data_source={args.data_source}",
            f"start_date={args.start_date}",
            f"end_date={args.end_date}",
            f"franchise={args.franchise}",
            f"grade={args.grade}",
            f"country_id={args.country_id}",
            f"entity_id={args.entity_id}",
        ]
        raise ValueError(
            "Query returned 0 rows. Verify filters and source data availability. "
            + " | ".join(details)
        )

    all_partners = get_unique_partners(df)
    run_output_dir = _get_versioned_output_dir(OUTPUT_DIR)
    
    if args.mode == "partner":
        partner_id = args.partner_id or all_partners[0]
        df_p = df[df["partner_id"] == partner_id].copy()
        result = process_partner(df_p, partner_id, df_p["partner_name"].iloc[0] if "partner_name" in df_p.columns else None)
        _save_partner_outputs(result, run_output_dir)
        _save_partner_outputs(result, OUTPUT_DIR)
        _write_latest_run_pointer(run_output_dir)
    else:
        cluster_results = run_franchise_mode(
            df,
            all_partners,
            run_output_dir,
            optimization_scope=args.optimization_scope,
        )
        _save_clustered_outputs(cluster_results, OUTPUT_DIR)
        _write_latest_run_pointer(run_output_dir)
        _log_franchise_optimal_summary(cluster_results, run_output_dir)


def _write_latest_run_pointer(run_output_dir: Path):
    """Persist a pointer to the latest run folder for quick discovery."""
    pointer_path = OUTPUT_DIR / "latest_run_path.txt"
    pointer_path.write_text(str(run_output_dir), encoding="utf-8")
    logger.info(f"Latest run pointer updated: {pointer_path}")
    logger.info(f"Latest run directory: {run_output_dir}")


def _log_franchise_optimal_summary(all_cluster_results, run_output_dir: Path):
    """Print a concise franchise optimization summary to the terminal."""
    if not all_cluster_results:
        return

    logger.info("============================================================")
    logger.info("FRANCHISE OPTIMAL CONFIG SUMMARY")
    logger.info("============================================================")

    global_result = next((res for res in all_cluster_results if res.get("cluster") == "Global"), None)
    if global_result:
        cfg = global_result.get("best_config", {})
        logger.info(
            "Global optimum | u1=%s | u2=%s | u3=%s | delta_ept=%s | duracion_hdm=%s | awt_improvement=%s | ept_increase=%s",
            cfg.get("u1"),
            cfg.get("u2"),
            cfg.get("u3"),
            cfg.get("delta_ept"),
            cfg.get("duracion_hdm"),
            cfg.get("awt_improvement"),
            cfg.get("ept_increase"),
        )

    for res in all_cluster_results:
        cfg = res.get("best_config", {})
        logger.info(
            "Cluster %s | u1=%s | u2=%s | u3=%s | delta_ept=%s | duracion_hdm=%s",
            res.get("cluster"),
            cfg.get("u1"),
            cfg.get("u2"),
            cfg.get("u3"),
            cfg.get("delta_ept"),
            cfg.get("duracion_hdm"),
        )

    logger.info(f"Detailed config CSV: {run_output_dir / 'franchise_optimal_config.csv'}")


def _save_partner_outputs(result, output_dir):
    """Save single-partner outputs (legacy format)."""
    optimizer = result["optimizer"]
    best_config = result["best_config"]

    history_path = output_dir / "optimization_history.csv"
    pd.DataFrame(optimizer.optimization_history).to_csv(history_path, index=False)
    logger.info(f"Saved optimization history: {history_path}")

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
    recommendations_path = output_dir / "hdm_recommendations.csv"
    pd.DataFrame([rec]).to_csv(recommendations_path, index=False)
    logger.info(f"Saved partner recommendations: {recommendations_path}")

    try:
        result["simulator"].generate_stress_day_analysis(
            result["data"],
            best_config["u1"],
            best_config["u2"],
            best_config["u3"],
            best_config["delta_ept"],
            best_config["duracion_hdm"],
            output_dir,
        )
    except Exception as e:
        logger.warning(f"Validation failed: {e}")


def _save_clustered_outputs(all_cluster_results, output_dir):
    """Save clustered franchise outputs."""
    all_history = []
    all_configs = []
    all_partner_results = []

    for res in all_cluster_results:
        cluster_name = res["cluster"]
        optimizer = res["optimizer"]
        best_config = res["best_config"]
        franchise_eval = res["evaluation"]

        hist = pd.DataFrame(optimizer.optimization_history)
        hist["cluster"] = cluster_name
        all_history.append(hist)

        all_configs.append(best_config)

        p_res = pd.DataFrame(franchise_eval["partner_results"])
        p_res["cluster"] = cluster_name
        all_partner_results.append(p_res)

    history_path = output_dir / "optimization_history.csv"
    config_path = output_dir / "franchise_optimal_config.csv"
    partner_impact_path = output_dir / "franchise_impact_by_partner.csv"

    pd.concat(all_history).to_csv(history_path, index=False)
    pd.DataFrame(all_configs).to_csv(config_path, index=False)
    pd.concat(all_partner_results).to_csv(partner_impact_path, index=False)

    logger.info(f"Saved optimization history: {history_path}")
    logger.info(f"Saved franchise optimal config: {config_path}")
    logger.info(f"Saved franchise impact by partner: {partner_impact_path}")

    logger.info(f"Clustered optimization complete. Clusters saved to {output_dir}")

if __name__ == "__main__":
    main()
