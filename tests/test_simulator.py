import unittest
import pandas as pd
import numpy as np
from src.simulator import HDMSimulator

class MockPredictor:
    def predict(self, ordenes, riders, hdm, ept):
        return 10.0 # Constant AWT for testing

class TestHDMSimulator(unittest.TestCase):
    def setUp(self):
        self.baseline_metrics = {"awt_promedio": 10.0, "ept_promedio": 20.0}
        self.simulator = HDMSimulator(MockPredictor(), MockPredictor(), self.baseline_metrics)

    def test_should_activate_hdm(self):
        # All conditions met
        self.assertTrue(self.simulator.should_activate_hdm(5, 2, 5, u1=5, u2=2, u3=5))
        # One condition fails
        self.assertFalse(self.simulator.should_activate_hdm(4, 2, 5, u1=5, u2=2, u3=5))
        self.assertFalse(self.simulator.should_activate_hdm(5, 1, 5, u1=5, u2=2, u3=5))
        self.assertFalse(self.simulator.should_activate_hdm(5, 2, 4, u1=5, u2=2, u3=5))

    def test_simulation_loop_basic(self):
        # Create a tiny dataframe with minutes far apart to clearly test delay and active periods
        df = pd.DataFrame({
            "momento_exacto": pd.to_datetime([
                "2026-01-01 10:00:00",
                "2026-01-01 10:01:00",
                "2026-01-01 10:02:00",
                "2026-01-01 10:03:00"
            ]),
            "ordenes_pendientes": [10, 10, 10, 10],
            "riders_cerca": [5, 5, 5, 5],
            "max_awt_espera_min": [10, 10, 10, 10],
            "ept_promedio": [20, 20, 20, 20],
            "hdm_activo": [0, 0, 0, 0]
        })

        # Test activation with delay.
        # ACTIVATION_DELAY_MINUTES is 2 in config.py
        # T=0: 10:00:00 -> Triggered. activation_queue_start = 10:00:00. in_delay = 1.
        # T=1: 10:01:00 -> curr < 10:00:00 + 2min (10:02:00). in_delay = 1.
        # T=2: 10:02:00 -> curr == 10:00:00 + 2min. Delay expired. hdm_active = 1.

        res = self.simulator._run_simulation_loop(df, u1=5, u2=2, u3=5, delta_ept=5, duracion_hdm=10)

        self.assertEqual(res["hdm_activated_sim"].iloc[0], 1)
        self.assertEqual(res["hdm_in_delay_sim"].iloc[0], 1)
        self.assertEqual(res["hdm_in_delay_sim"].iloc[1], 1)
        self.assertEqual(res["hdm_active_sim"].iloc[0], 0)
        # At T=2, delay has passed (exactly 2 minutes)
        self.assertEqual(res["hdm_active_sim"].iloc[2], 1)
        self.assertEqual(res["ept_with_hdm"].iloc[2], 25)

if __name__ == '__main__':
    unittest.main()
