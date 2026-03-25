"""Hourly average profile artifacts by day-of-week (Mon-Sun) for global franchise recommendations."""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .config import TRAIN_TEST_SPLIT
from .model import temporal_train_test_split_df
from .simulator import HDMSimulator

logger = logging.getLogger(__name__)

TIMELINE_CSV_NAME = "test_timeline_global_equilibrada.csv"

# pandas dt.dayofweek: Mon=0 ... Sun=6
_DAY_OF_WEEK_MAP = {
    0: {"label": "Lunes", "slug": "lunes"},
    1: {"label": "Martes", "slug": "martes"},
    2: {"label": "Miercoles", "slug": "miercoles"},
    3: {"label": "Jueves", "slug": "jueves"},
    4: {"label": "Viernes", "slug": "viernes"},
    5: {"label": "Sabado",  "slug": "sabado"},
    6: {"label": "Domingo", "slug": "domingo"},
}

_METRIC_SPECS = [
    ("awt",     "awt_real",     "awt_sim",     "AWT (min)"),
    ("ept",     "ept_real",     "ept_sim",     "EPT (min)"),
    ("hdm",     "hdm_real",     "hdm_sim",     "HDM activo (proporción)"),
    ("ordenes", "ordenes_real", "ordenes_sim", "Ordenes pendientes (prom.)"),
    ("riders",  "riders_real",  "riders_sim",  "Riders esperando (prom.)"),
]


def _aggregate_hourly_by_dayofweek(df_sim: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Aggregate simulation output to average hourly profile per day-of-week.

    For each day in _DAY_OF_WEEK_MAP (Mon=0 ... Sun=6) returns a DataFrame
    with one row per hour (0-23) containing the mean of each metric across all
    historical occurrences of that day in the dataset.

    HDM columns are kept as a fraction (0.0–1.0): proportion of observations
    where HDM was active at that hour — more informative than a hard 0/1 average.
    """
    df = df_sim.copy()
    ts = pd.to_datetime(df["momento_exacto"])
    df["_dow"] = ts.dt.dayofweek
    df["_hour"] = ts.dt.hour

    # First collapse to one row per (date, partner, hour) so each calendar day
    # contributes equally when averaging across weeks.
    df["_date"] = ts.dt.date
    per_slot = (
        df.groupby(["_dow", "_date", "_hour"], as_index=False)
        .agg(
            awt_real=("max_awt_espera_min",  "mean"),
            awt_sim=("awt_predicted",        "mean"),
            ept_real=("ept_base",            "mean"),
            ept_sim=("ept_with_hdm",         "mean"),
            hdm_real=("hdm_activo",          "mean"),   # fraction 0-1
            hdm_sim=("hdm_active_sim",       "mean"),   # fraction 0-1
            ordenes_real=("ordenes_pendientes", "mean"),
            ordenes_sim=("ordenes_pendientes",  "mean"),
            riders_real=("riders_cerca",     "mean"),
            riders_sim=("riders_cerca",      "mean"),
        )
    )

    result: Dict[int, pd.DataFrame] = {}
    for dow in _DAY_OF_WEEK_MAP:
        df_day = per_slot[per_slot["_dow"] == dow]
        if df_day.empty:
            logger.warning("No test data found for day-of-week=%d (%s).", dow, _DAY_OF_WEEK_MAP[dow]["label"])
            continue
        hourly = (
            df_day.groupby("_hour", as_index=False)
            .agg(
                awt_real=("awt_real",     "mean"),
                awt_sim=("awt_sim",       "mean"),
                ept_real=("ept_real",     "mean"),
                ept_sim=("ept_sim",       "mean"),
                hdm_real=("hdm_real",     "mean"),
                hdm_sim=("hdm_sim",       "mean"),
                ordenes_real=("ordenes_real", "mean"),
                ordenes_sim=("ordenes_sim",   "mean"),
                riders_real=("riders_real",   "mean"),
                riders_sim=("riders_sim",     "mean"),
            )
            .rename(columns={"_hour": "hora"})
            .sort_values("hora")
            .reset_index(drop=True)
        )
        result[dow] = hourly
    return result


def _plot_hourly_profile(
    df_hourly: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot average hourly profile for one day-of-week across 5 metric panels."""
    fig, axes = plt.subplots(len(_METRIC_SPECS), 1, figsize=(14, 16), sharex=True)

    hours = df_hourly["hora"]
    for idx, (metric_key, real_col, sim_col, ylabel) in enumerate(_METRIC_SPECS):
        ax = axes[idx]
        ax.plot(hours, df_hourly[real_col],  label="Real",        color="#1f77b4", linewidth=1.8, marker="o", markersize=3)
        ax.plot(hours, df_hourly[sim_col],   label="Recomendado", color="#ff7f0e", linewidth=1.8, marker="o", markersize=3)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        if metric_key == "hdm":
            ax.set_ylim(-0.05, 1.1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    axes[0].legend(loc="upper right", ncol=2, fontsize=10)
    axes[-1].set_xlabel("Hora del día")
    axes[-1].set_xticks(range(0, 24))
    axes[-1].set_xticklabels([f"{h:02d}h" for h in range(24)], rotation=45, ha="right", fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _get_required_config(best_config: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = ["u1", "u2", "u3", "delta_ept", "duracion_hdm"]
    missing = [key for key in required_keys if key not in best_config]
    if missing:
        raise ValueError(f"Missing required config keys for timeline report: {missing}")
    return {
        "u1": best_config["u1"],
        "u2": best_config["u2"],
        "u3": best_config["u3"],
        "delta_ept": best_config["delta_ept"],
        "duracion_hdm": best_config["duracion_hdm"],
    }


def generate_global_test_timeline_artifacts(
    df_global: pd.DataFrame,
    awt_predictor,
    ept_predictor,
    baseline_metrics: Dict[str, Any],
    best_config: Dict[str, Any],
    output_dirs: Iterable[Path],
    rolling_window_minutes: int | None = None,  # kept for API compatibility, not used
) -> Dict[str, Any]:
    """
    Build and persist average hourly profile artifacts per day-of-week (Mon-Sun).

    Outputs per output_dir:
    - test_timeline_global_equilibrada.csv  (all days, columns: day_label, hora, metrics)
    - global_test_timeline_<day_slug>.png (one PNG per available day in test slice)
    """
    _, df_test, split_metadata = temporal_train_test_split_df(df_global, TRAIN_TEST_SPLIT)
    if df_test.empty:
        msg = f"Skipping timeline artifacts: temporal test slice is empty (total rows={len(df_global)}, split={TRAIN_TEST_SPLIT})."
        logger.warning(msg)
        print(f"\n[TIMELINE] {msg}\n", flush=True)
        return {"saved": False, "reason": "empty_test_slice", "split": split_metadata}

    cfg = _get_required_config(best_config)
    simulator = HDMSimulator(awt_predictor, ept_predictor, baseline_metrics)
    df_sim_test = simulator.simulate_timeline(
        df_test,
        cfg["u1"],
        cfg["u2"],
        cfg["u3"],
        cfg["delta_ept"],
        cfg["duracion_hdm"],
    )

    hourly_by_dow = _aggregate_hourly_by_dayofweek(df_sim_test)
    if not hourly_by_dow:
        msg = "Skipping timeline artifacts: no day-of-week data found in test slice."
        logger.warning(msg)
        print(f"\n[TIMELINE] {msg}\n", flush=True)
        return {"saved": False, "reason": "no_day_data", "split": split_metadata}

    # Build combined CSV (all days stacked)
    csv_frames = []
    for dow, df_h in hourly_by_dow.items():
        df_h = df_h.copy()
        df_h.insert(0, "dia", _DAY_OF_WEEK_MAP[dow]["label"])
        csv_frames.append(df_h)
    df_csv = pd.concat(csv_frames, ignore_index=True)

    test_share_pct = split_metadata["test_ratio"] * 100.0
    title_base = f"Global Equilibrada — Perfil horario promedio | Test ({test_share_pct:.0f}% del dataset)"

    saved_paths: Dict[str, Dict[str, str]] = {}
    for out_dir in output_dirs:
        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_path = target_dir / TIMELINE_CSV_NAME
        df_csv.to_csv(csv_path, index=False)

        png_paths: Dict[str, str] = {"csv": str(csv_path)}
        for dow, df_h in hourly_by_dow.items():
            day_info = _DAY_OF_WEEK_MAP[dow]
            png_name = f"global_test_timeline_{day_info['slug']}.png"
            png_path = target_dir / png_name
            _plot_hourly_profile(
                df_h,
                png_path,
                f"{title_base} | {day_info['label']}",
            )
            png_paths[day_info["slug"]] = str(png_path)

        saved_paths[str(target_dir)] = png_paths

    days_generated = [_DAY_OF_WEEK_MAP[d]["label"] for d in hourly_by_dow]
    logger.info(
        "Saved hourly profile artifacts for %s (%s output directories).",
        ", ".join(days_generated),
        len(saved_paths),
    )
    return {
        "saved": True,
        "split": split_metadata,
        "days": days_generated,
        "paths": saved_paths,
    }
