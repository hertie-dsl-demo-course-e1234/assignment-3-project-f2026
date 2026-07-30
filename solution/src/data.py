"""Synthetic data for the model-answer project.

Six hundred municipality-years. The generating process, stated openly so the report can be
honest about it:

    missed_target ~ Bernoulli(sigmoid(eta))
    eta = 1.2*deprivation_z - 1.0*collection_points_z - 0.5*current_rate_z
          + 0.4*pop_density_z + region_effect + noise - 1.1

Municipalities are nested in twelve regions with their own effects, which is what makes a
grouped split (rather than a random one) the honest choice - see REPORT.md section 3.

Two columns carry missing values, and the missingness is informative: a council that fails
to report its collection points is more likely to miss the target, so the imputation
indicator is a real feature rather than bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 2026
N = 600
REGIONS = [f"R{i:02d}" for i in range(1, 13)]


def _z(values: np.ndarray) -> np.ndarray:
    return (values - np.nanmean(values)) / np.nanstd(values)


def load_data() -> pd.DataFrame:
    """Return the full frame, including the `missed_target` column and `region` group key."""
    rng = np.random.default_rng(SEED)

    region = rng.choice(REGIONS, N)
    region_effect = dict(zip(REGIONS, rng.normal(0, 0.6, len(REGIONS))))

    pop_density = rng.lognormal(6.2, 0.7, N).round(1)          # people per km2
    deprivation = rng.normal(50, 12, N).round(1)               # index, higher = worse
    collection_points = rng.poisson(14, N).astype(float)
    # Last year's recycling rate: correlated with the drivers, but measured noisily -
    # which is exactly why ranking on it (the authority's current practice) is a weak
    # baseline rather than a hopeless one.
    current_rate = np.clip(
        58 - 0.10 * (deprivation - 50) + 0.20 * collection_points + rng.normal(0, 9, N),
        20, 85).round(1)
    kerbside_weeks = rng.choice([1, 2, 4], N, p=[0.45, 0.4, 0.15]).astype(float)
    collection_type = rng.choice(["kerbside", "mixed", "bring_bank"], N,
                                 p=[0.55, 0.3, 0.15])

    eta = (1.2 * _z(deprivation) - 1.0 * _z(collection_points)
           - 0.5 * _z(current_rate) + 0.4 * _z(pop_density)
           + np.array([region_effect[r] for r in region])
           + rng.normal(0, 0.8, N) - 1.1)
    missed = rng.random(N) < 1 / (1 + np.exp(-eta))

    frame = pd.DataFrame({
        "region": region,
        "collection_type": collection_type,
        "pop_density": pop_density,
        "deprivation_index": deprivation,
        "collection_points": collection_points,
        "current_rate": current_rate,
        "kerbside_weeks": kerbside_weeks,
        "missed_target": missed.astype(int),
    })

    # Informative missingness: non-reporting is commoner among councils that miss.
    p_missing_points = np.where(frame.missed_target == 1, 0.11, 0.05)
    frame.loc[rng.random(N) < p_missing_points, "collection_points"] = np.nan
    frame.loc[rng.random(N) < 0.04, "deprivation_index"] = np.nan

    return frame


def region_size_tercile(frame: pd.DataFrame) -> pd.Series:
    """Small / medium / large by population density - the subgroup the report disaggregates."""
    return pd.qcut(frame.pop_density, 3, labels=["small", "medium", "large"])


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(f"\nn = {len(df)}, base rate = {df.missed_target.mean():.3f}")
    print(df.isna().sum().rename("missing").to_frame().T)
