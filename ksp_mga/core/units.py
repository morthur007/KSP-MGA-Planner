"""Unit constants used throughout the pipeline.

Canonical internal convention by layer:
  - SPICE/PyKEP/Lambert: km, km/s, seconds ET
  - Principia native daemon: m, m/s, seconds ET
"""

DAY_S = 86400.0
KM_TO_M = 1000.0
M_TO_KM = 0.001


def days_to_seconds(days: float) -> float:
    return float(days) * DAY_S


def seconds_to_days(seconds: float) -> float:
    return float(seconds) / DAY_S
