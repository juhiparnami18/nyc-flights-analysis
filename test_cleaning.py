"""Run with:  python -m unittest discover tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cleaning import classify_status, clean_weather, hhmm_to_minutes, iqr_bounds  # noqa: E402


class TestTimeConversion(unittest.TestCase):
    def test_known_values(self):
        out = hhmm_to_minutes(pd.Series([517, 1230, 2359, 5]))
        self.assertEqual(out.tolist(), [317, 750, 1439, 5])

    def test_midnight_and_missing(self):
        out = hhmm_to_minutes(pd.Series([2400, np.nan]))
        self.assertEqual(out.iloc[0], 0)
        self.assertTrue(np.isnan(out.iloc[1]))


class TestStatus(unittest.TestCase):
    def test_all_four_statuses(self):
        df = pd.DataFrame({
            "dep_time": [np.nan, 900, 900, 900],
            "arr_time": [np.nan, np.nan, 1100, 1100],
            "arr_delay": [np.nan, np.nan, np.nan, 5.0],
        })
        self.assertEqual(classify_status(df).tolist(),
                         ["Cancelled", "Diverted", "Delay not recorded", "Completed"])


class TestWeather(unittest.TestCase):
    def test_impossible_wind_removed_and_gap_filled(self):
        w = pd.DataFrame({
            "origin": ["JFK"] * 4,
            "time_hour": ["2013-01-01T00:00:00Z", "2013-01-01T01:00:00Z",
                          "2013-01-01T02:00:00Z", "2013-01-01T03:00:00Z"],
            "temp": [30.0, np.nan, 34.0, 36.0], "dewp": [20.0] * 4, "humid": [50.0] * 4,
            "wind_speed": [10.0, 1048.0, 12.0, 14.0], "wind_dir": [90.0] * 4, "pressure": [1010.0] * 4,
        })
        cleaned, log = clean_weather(w)
        self.assertEqual(log["impossible_wind"], 1)
        self.assertAlmostEqual(cleaned.loc[1, "wind_speed"], 11.0)   # interpolated between 10 and 12
        self.assertAlmostEqual(cleaned.loc[1, "temp"], 32.0)
        self.assertEqual(int(cleaned["wind_speed"].isna().sum()), 0)


class TestIQR(unittest.TestCase):
    def test_bounds(self):
        low, high = iqr_bounds(pd.Series([1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertAlmostEqual(low, -2.5)
        self.assertAlmostEqual(high, 11.5)


if __name__ == "__main__":
    unittest.main()
