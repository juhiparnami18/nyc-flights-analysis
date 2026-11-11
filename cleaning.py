"""Small, tested helpers used by the analysis notebook.

Keeping the logic here (instead of burying it in notebook cells) makes it easy to
test and reuse, which is one of the core industry best practices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """Convert clock times stored as integers (hhmm) into minutes after midnight.

    Example: 517 -> 5:17 -> 317 minutes. The value 2400 means midnight, so it becomes 0.
    Missing values stay missing.
    """
    s = series.astype("float64")
    s = s.where(s != 2400, 0)
    return (s // 100) * 60 + (s % 100)


def classify_status(df: pd.DataFrame) -> pd.Series:
    """Label every flight with what actually happened to it.

    - Cancelled: it never departed (no departure time)
    - Diverted: it departed but never reached the planned destination (no arrival time)
    - Delay not recorded: it arrived but the arrival delay is missing
    - Completed: it departed and arrived with a recorded delay
    """
    status = np.select(
        [df["dep_time"].isna(), df["arr_time"].isna(), df["arr_delay"].isna()],
        ["Cancelled", "Diverted", "Delay not recorded"],
        default="Completed",
    )
    return pd.Series(status, index=df.index, name="status")


def clean_weather(
    weather: pd.DataFrame, max_wind_mph: float = 100.0, interp_limit: int = 6
) -> tuple[pd.DataFrame, dict]:
    """Fix impossible wind speeds and fill short hourly gaps.

    Returns the cleaned table and a small dictionary that counts every change.
    """
    w = weather.copy()
    w["time_hour"] = pd.to_datetime(w["time_hour"], utc=True)
    w = w.sort_values(["origin", "time_hour"]).reset_index(drop=True)

    log = {"impossible_wind": int((w["wind_speed"] > max_wind_mph).sum())}
    w.loc[w["wind_speed"] > max_wind_mph, "wind_speed"] = np.nan

    fill_cols = ["temp", "dewp", "humid", "wind_speed", "wind_dir", "pressure"]
    log["missing_before"] = {c: int(w[c].isna().sum()) for c in fill_cols}
    for c in fill_cols:
        w[c] = w.groupby("origin")[c].transform(
            lambda s: s.interpolate(limit=interp_limit, limit_direction="both")
        )
    log["missing_after"] = {c: int(w[c].isna().sum()) for c in fill_cols}
    return w, log


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences: values outside [Q1 - k*IQR, Q3 + k*IQR] are statistical outliers."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)
