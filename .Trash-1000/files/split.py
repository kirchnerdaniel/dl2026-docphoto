import numpy as np
import pandas as pd


def build_splits(
    df: pd.DataFrame,
    group_column: str,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-based train/validation split with reproducible results."""

    if group_column not in df.columns:
        raise ValueError(f"Missing group column: {group_column}")

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if df[group_column].isna().any():
        raise ValueError(f"Missing values in group column: {group_column}")

    groups = df[group_column].unique()

    if len(groups) < 2:
        raise ValueError("At least two unique groups are required.")

    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)

    n_val = int(round(len(groups) * val_fraction))
    n_val = max(1, min(n_val, len(groups) - 1))

    val_groups = groups[:n_val]
    val_mask = df[group_column].isin(val_groups).to_numpy()

    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)

    return train_idx, val_idx