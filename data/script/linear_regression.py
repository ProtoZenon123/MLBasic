"""Generate reproducible linear-regression datasets with and without outliers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "generated_data"
TRUE_WEIGHTS = np.array([1.5, -2.0, 0.5, 3.0, -1.0, 2.5, -0.75, 1.25, -3.5, 0.8])
FEATURE_NAMES = [f"feat_{i}" for i in range(1, len(TRUE_WEIGHTS) + 1)]


def generate_datasets(
    n_samples: int = 100,
    n_outliers: int = 8,
    seed: int = 42,
    intercept: float = 1.25,
    noise_std: float = 1.0,
    outlier_shift: float = 35.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a noisy 10-feature dataset and a copy with response outliers.

    Both datasets share features and ordinary Gaussian noise. Outliers have their
    target values shifted upward at large feat_1 values.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if not 0 <= n_outliers <= n_samples:
        raise ValueError("n_outliers must be between 0 and n_samples")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if outlier_shift <= 0:
        raise ValueError("outlier_shift must be positive")

    rng = np.random.default_rng(seed)
    feat_matrix = rng.uniform(-10.0, 10.0, size=(n_samples, len(TRUE_WEIGHTS)))
    target_values = (
        feat_matrix @ TRUE_WEIGHTS
        + intercept
        + rng.normal(0.0, noise_std, size=n_samples)
    )

    clean = pd.DataFrame(feat_matrix, columns=FEATURE_NAMES)
    clean["target"] = target_values
    clean["is_outlier"] = False
    with_outliers = clean.copy()

    # Choose large feat_1 values so outliers affect its fitted weight.
    outlier_indices = np.argsort(feat_matrix[:, 0])[-n_outliers:] if n_outliers else []
    with_outliers.loc[outlier_indices, "target"] += outlier_shift
    with_outliers.loc[outlier_indices, "is_outlier"] = True

    return clean, with_outliers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--outliers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7523)
    parser.add_argument("--intercept", type=float, default=1.25)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--outlier-shift", type=float, default=35.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    clean, with_outliers = generate_datasets(
        n_samples=args.samples,
        n_outliers=args.outliers,
        seed=args.seed,
        intercept=args.intercept,
        noise_std=args.noise_std,
        outlier_shift=args.outlier_shift,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = args.output_dir / "linear_regression_clean.csv"
    outlier_path = args.output_dir / "linear_regression_outliers.csv"
    clean.to_csv(clean_path, index=False)
    with_outliers.to_csv(outlier_path, index=False)
    print(f"Generated {len(clean)} rows in each dataset:")
    print(f"  {clean_path}")
    print(f"  {outlier_path} ({args.outliers} outliers)")
    print(f"True parameters (w0 bias, w1..w10): {[args.intercept, *TRUE_WEIGHTS.tolist()]}")
    print(f"Noise std: {args.noise_std}")


if __name__ == "__main__":
    main()
