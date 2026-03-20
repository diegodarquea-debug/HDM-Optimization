"""Generate minute-level test timeline artifacts for global franchise recommendations."""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .config import TIMELINE_REPORT_SETTINGS, TRAIN_TEST_SPLIT
from .model import temporal_train_test_split_df
from .simulator import HDMSimulator

logger = logging.getLogger(__name__)

TIMELINE_CSV_NAME = "test_timeline_global_equilibrada.csv"
TIMELINE_RAW_PNG_NAME = "global_test_timeline_raw.png"
TIMELINE_SMOOTHED_PNG_NAME = "global_test_timeline_smoothed.png"

_METRIC_SPECS = [
    ("awt", "awt_real", "awt_sim", "AWT (min)"),
    ("ept", "ept_real", "ept_sim", "EPT (min)"),
    ("hdm", "hdm_real", "hdm_sim", "HDM activo (0=off, 1=on)"),
    ("ordenes", "ordenes_real", "ordenes_sim", "Ordenes pendientes"),
    ("riders", "riders_real", "riders_sim", "Riders esperando"),
]


def _aggregate_minute_level(df_sim: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level simulation output to one row per minute."""
    timeline = (
        df_sim.groupby("momento_exacto", as_index=False)
        .agg(
            awt_real=("max_awt_espera_min", "mean"),
            awt_sim=("awt_predicted", "mean"),
            ept_real=("ept_base", "mean"),
            ept_sim=("ept_with_hdm", "mean"),
            hdm_real=("hdm_activo", "max"),
            hdm_sim=("hdm_active_sim", "max"),
            ordenes_real=("ordenes_pendientes", "sum"),
            ordenes_sim=("ordenes_pendientes", "sum"),
            riders_real=("riders_cerca", "sum"),
            riders_sim=("riders_cerca", "sum"),
        )
        .sort_values("momento_exacto")
        .reset_index(drop=True)
    )

    # HDM is binary: 1 if active in any row of that minute, 0 otherwise.
    timeline["hdm_real"] = timeline["hdm_real"].clip(0, 1).astype(int)
    timeline["hdm_sim"] = timeline["hdm_sim"].clip(0, 1).astype(int)
    return timeline


def _smooth_timeline(df_timeline: pd.DataFrame, rolling_window_minutes: int) -> pd.DataFrame:
    """Return a rolling-smoothed copy of the minute-level timeline."""
    df_smooth = df_timeline.copy()
    value_cols = [col for col in df_smooth.columns if col != "momento_exacto"]
    for col in value_cols:
        df_smooth[col] = df_smooth[col].rolling(window=rolling_window_minutes, min_periods=1).mean()
    return df_smooth


def _plot_timeline(
    df_timeline: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot the five requested metrics as aligned panels over time."""
    fig, axes = plt.subplots(len(_METRIC_SPECS), 1, figsize=(18, 16), sharex=True)

    for idx, (metric_key, real_col, sim_col, ylabel) in enumerate(_METRIC_SPECS):
        ax = axes[idx]
        ax.plot(df_timeline["momento_exacto"], df_timeline[real_col], label="Real", color="#1f77b4", linewidth=1.1)
        ax.plot(df_timeline["momento_exacto"], df_timeline[sim_col], label="Recomendado", color="#ff7f0e", linewidth=1.1)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if metric_key == "hdm":
            ax.set_ylim(-0.1, 1.3)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0 (off)", "1 (on)"])

    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Fecha (nivel minuto)")
    fig.suptitle(title, fontsize=14)
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
    rolling_window_minutes: int | None = None,
) -> Dict[str, Any]:
    """
    Build and persist global test-only minute-level comparison artifacts.

    Outputs:
    - test_timeline_global_equilibrada.csv
    - global_test_timeline_raw.png
    - global_test_timeline_smoothed.png
    """
    rolling_window = rolling_window_minutes or TIMELINE_REPORT_SETTINGS["rolling_window_minutes"]

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

    df_timeline_raw = _aggregate_minute_level(df_sim_test)
    df_timeline_smooth = _smooth_timeline(df_timeline_raw, rolling_window_minutes=rolling_window)

    test_share_pct = split_metadata["test_ratio"] * 100.0
    title_base = f"Global Equilibrada | Test temporal final ({test_share_pct:.0f}% del dataset)"

    saved_paths: Dict[str, Dict[str, str]] = {}
    for out_dir in output_dirs:
        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_path = target_dir / TIMELINE_CSV_NAME
        raw_png_path = target_dir / TIMELINE_RAW_PNG_NAME
        smooth_png_path = target_dir / TIMELINE_SMOOTHED_PNG_NAME

        df_timeline_raw.to_csv(csv_path, index=False)
        _plot_timeline(df_timeline_raw, raw_png_path, f"{title_base} | Serie cruda")
        _plot_timeline(
            df_timeline_smooth,
            smooth_png_path,
            f"{title_base} | Serie suavizada (rolling {rolling_window} min)",
        )

        saved_paths[str(target_dir)] = {
            "csv": str(csv_path),
            "raw_png": str(raw_png_path),
            "smoothed_png": str(smooth_png_path),
        }

    logger.info("Saved global test timeline artifacts for %s output directories.", len(saved_paths))
    return {
        "saved": True,
        "split": split_metadata,
        "rolling_window_minutes": rolling_window,
        "paths": saved_paths,
    }
