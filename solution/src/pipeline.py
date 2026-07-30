"""Project pipeline - the model answer's single entry point.

    python -m src.pipeline

Load, split (grouped by region), tune on the training regions only, score the held-out
regions once, and report every number that appears in REPORT.md - with a baseline, an
interval, and a subgroup breakdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import load_data, region_size_tercile

SEED = 2026
N_VISITS = 20  # the authority funds twenty advisory visits a year - hence precision@20

NUM_COLS = ["pop_density", "deprivation_index", "collection_points",
            "current_rate", "kerbside_weeks"]
CAT_COLS = ["collection_type"]


def build_preprocessor() -> ColumnTransformer:
    """Every fitted transformation lives in here, so cross-validation cannot leak."""
    return ColumnTransformer([
        ("num", Pipeline([
            # add_indicator keeps "this was not reported", which is informative here
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


def build_models() -> dict[str, tuple[Pipeline, dict]]:
    """The interpretable candidate and the flexible one, with their tuning grids."""
    return {
        "logistic": (
            Pipeline([("pre", build_preprocessor()),
                      ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": np.logspace(-2, 2, 9)},
        ),
        "boosting": (
            Pipeline([("pre", build_preprocessor()),
                      ("clf", HistGradientBoostingClassifier(random_state=SEED))]),
            {"clf__learning_rate": [0.03, 0.1, 0.3]},
        ),
    }


def split(frame: pd.DataFrame):
    """A GROUPED hold-out: whole regions go to test, never individual municipalities.

    Regional policy is a shared cause, so splitting municipalities at random would let the
    model memorise region effects and report a score it could not repeat on a new region.
    """
    rng = np.random.default_rng(SEED)
    regions = np.sort(frame.region.unique())
    test_regions = set(rng.choice(regions, size=max(1, round(0.3 * len(regions))),
                                  replace=False))
    is_test = frame.region.isin(test_regions)
    return frame.loc[~is_test].copy(), frame.loc[is_test].copy(), sorted(test_regions)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Of the k highest-scored municipalities, the share that did miss the target."""
    k = min(k, len(scores))
    top = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[top]))


def bootstrap_ci(y_true, scores, metric, n_boot=2000, level=0.95, seed=SEED):
    """Percentile bootstrap over the test rows - a point estimate is not an estimate."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric(y_true[idx], scores[idx]))
    values = np.sort(values)
    return (float(values[int((1 - level) / 2 * len(values))]),
            float(values[int((1 + level) / 2 * len(values)) - 1]))


def evaluate(name, y_true, scores, k=N_VISITS):
    auc = roc_auc_score(y_true, scores)
    lo, hi = bootstrap_ci(y_true, scores, roc_auc_score)
    return {"model": name, "auc": auc, "ci_low": lo, "ci_high": hi,
            "precision_at_k": precision_at_k(y_true, scores, k)}


def run_pipeline() -> dict:
    """Load, split, tune, score once, and return every number the report quotes."""
    frame = load_data()
    train, test, test_regions = split(frame)
    y_tr = train.pop("missed_target").to_numpy()
    y_te = test.pop("missed_target").to_numpy()
    groups = train.region.to_numpy()

    print(f"n = {len(frame)}, base rate = {(y_tr.sum() + y_te.sum()) / len(frame):.3f}")
    print(f"train {len(y_tr)} rows / {train.region.nunique()} regions, "
          f"test {len(y_te)} rows / {len(test_regions)} regions {test_regions}")

    results = []

    # Baseline: what the authority does today - rank by last year's recycling rate,
    # ascending (a low rate is high risk). Uses no model at all.
    baseline_scores = -test.current_rate.to_numpy()
    results.append(evaluate("baseline: rank by current rate", y_te, baseline_scores))

    cv = GroupKFold(n_splits=5)
    fitted = {}
    for name, (pipe, grid) in build_models().items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1)
        search.fit(train, y_tr, groups=groups)   # groups: tuning respects the regions too
        scores = search.predict_proba(test)[:, 1]
        fitted[name] = (search, scores)
        row = evaluate(name, y_te, scores)
        row["best_params"] = search.best_params_
        row["cv_auc"] = search.best_score_
        results.append(row)

    print(f"\n{'model':<32} {'test AUC':>9} {'95% CI':>18} {'prec@20':>9}")
    for row in results:
        ci = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        print(f"{row['model']:<32} {row['auc']:>9.3f} {ci:>18} {row['precision_at_k']:>9.2f}")

    # Subgroup: the aggregate is an average over municipalities, so disaggregate.
    chosen, chosen_scores = fitted["logistic"]
    tercile = region_size_tercile(frame).loc[test.index]
    print(f"\nlogistic model by size tercile:\n{'group':<8} {'n':>4} {'AUC':>7}   95% CI")
    subgroups = {}
    for level in ("small", "medium", "large"):
        mask = (tercile == level).to_numpy()
        if mask.sum() < 20 or len(np.unique(y_te[mask])) < 2:
            continue
        auc = roc_auc_score(y_te[mask], chosen_scores[mask])
        lo, hi = bootstrap_ci(y_te[mask], chosen_scores[mask], roc_auc_score, seed=7)
        subgroups[level] = {"n": int(mask.sum()), "auc": auc, "ci": (lo, hi)}
        print(f"{level:<8} {mask.sum():>4} {auc:>7.3f}   [{lo:.3f}, {hi:.3f}]")

    print(f"\nchosen model: logistic ({chosen.best_params_}); "
          f"reported because its interval overlaps boosting's and it is interpretable.")
    return {"results": results, "subgroups": subgroups,
            "test_regions": test_regions, "seed": SEED}


if __name__ == "__main__":
    run_pipeline()
