"""Shared helpers for Cowtrol Systems."""
import os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def path(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def load_config(file="config.yaml"):
    with open(path(file)) as f:
        return yaml.safe_load(f)


def nrc_thi(temp_c, rh_pct):
    """NRC (1971) THI: (1.8T+32) - (0.55 - 0.0055*RH)(1.8T - 26), T in C, RH in %."""
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh_pct, dtype=float)
    return (1.8 * t + 32) - (0.55 - 0.0055 * rh) * (1.8 * t - 26)
