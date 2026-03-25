import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.timeline_reporting import (
    TIMELINE_CSV_NAME,
    generate_global_test_timeline_artifacts,
)


class _DummyModel:
    def predict(self, x):
        return np.full(shape=(len(x),), fill_value=5.0, dtype=float)


class _DummyAWTPredictor:
    def __init__(self):
        self.model = _DummyModel()
        self.ept_feature_name = None


class _DummyEPTPredictor:
    pass


class TestTimelineReporting(unittest.TestCase):
    def test_generate_global_test_timeline_artifacts(self):
        df = pd.DataFrame(
            {
                "momento_exacto": pd.date_range("2026-01-01 10:00:00", periods=10, freq="min"),
                "partner_id": [1] * 10,
                "ordenes_pendientes": [5, 6, 7, 8, 9, 10, 6, 5, 4, 3],
                "riders_cerca": [3, 3, 3, 4, 4, 4, 3, 2, 2, 2],
                "max_awt_espera_min": [8, 9, 10, 11, 12, 10, 9, 8, 7, 6],
                "ept_promedio_min": [14, 15, 16, 16, 17, 17, 16, 15, 14, 14],
                "hdm_activo": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            }
        )

        awt_predictor = _DummyAWTPredictor()
        ept_predictor = _DummyEPTPredictor()
        baseline_metrics = {
            "awt_promedio": float(df["max_awt_espera_min"].mean()),
            "ept_promedio_min_smoothed_promedio": float(df["ept_promedio_min"].mean()),
        }
        best_config = {
            "u1": 5,
            "u2": 2,
            "u3": 8,
            "delta_ept": 4,
            "duracion_hdm": 10,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            result = generate_global_test_timeline_artifacts(
                df_global=df,
                awt_predictor=awt_predictor,
                ept_predictor=ept_predictor,
                baseline_metrics=baseline_metrics,
                best_config=best_config,
                output_dirs=[out_dir],
                rolling_window_minutes=3,
            )

            self.assertTrue(result["saved"])

            csv_path = out_dir / TIMELINE_CSV_NAME
            png_paths = list(out_dir.glob("global_test_timeline_*.png"))

            self.assertTrue(csv_path.exists())
            self.assertGreaterEqual(len(png_paths), 1)

            df_timeline = pd.read_csv(csv_path)
            self.assertGreaterEqual(len(df_timeline), 1)
            self.assertIn("dia", df_timeline.columns)
            self.assertIn("awt_real", df_timeline.columns)
            self.assertIn("awt_sim", df_timeline.columns)
            self.assertIn("ept_real", df_timeline.columns)
            self.assertIn("ept_sim", df_timeline.columns)
            self.assertIn("hdm_real", df_timeline.columns)
            self.assertIn("hdm_sim", df_timeline.columns)
            self.assertIn("ordenes_real", df_timeline.columns)
            self.assertIn("riders_real", df_timeline.columns)
            self.assertIn("Jueves", set(df_timeline["dia"].dropna().unique()))


if __name__ == "__main__":
    unittest.main()
