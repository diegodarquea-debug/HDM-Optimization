import unittest
import pandas as pd

from src.model import temporal_train_test_split_df


class TestTemporalSplit(unittest.TestCase):
    def test_temporal_split_is_continuous_and_sorted(self):
        df = pd.DataFrame(
            {
                "momento_exacto": pd.to_datetime(
                    [
                        "2026-01-01 10:04:00",
                        "2026-01-01 10:00:00",
                        "2026-01-01 10:03:00",
                        "2026-01-01 10:01:00",
                        "2026-01-01 10:02:00",
                    ]
                ),
                "partner_id": [1, 1, 1, 1, 1],
                "ordenes_pendientes": [5, 4, 3, 2, 1],
            }
        )

        df_train, df_test, split = temporal_train_test_split_df(df, train_ratio=0.6)

        self.assertEqual(split["n_total"], 5)
        self.assertEqual(split["n_train"], 3)
        self.assertEqual(split["n_test"], 2)
        self.assertTrue(split["is_temporal"])

        self.assertTrue(df_train["momento_exacto"].is_monotonic_increasing)
        self.assertTrue(df_test["momento_exacto"].is_monotonic_increasing)

        self.assertLess(df_train["momento_exacto"].iloc[-1], df_test["momento_exacto"].iloc[0])

    def test_temporal_split_single_row(self):
        df = pd.DataFrame(
            {
                "momento_exacto": pd.to_datetime(["2026-01-01 10:00:00"]),
                "partner_id": [1],
                "ordenes_pendientes": [5],
            }
        )

        df_train, df_test, split = temporal_train_test_split_df(df, train_ratio=0.6)

        self.assertEqual(split["n_total"], 1)
        self.assertEqual(split["n_train"], 1)
        self.assertEqual(split["n_test"], 0)
        self.assertEqual(len(df_train), 1)
        self.assertEqual(len(df_test), 0)


if __name__ == "__main__":
    unittest.main()
