"""Stage 1: leakage-free ingestion and preprocessing for arbitrary tabular regression.

Numpy / scikit-learn / pandas only (no torch), so it can be tested on its own.

Order of operations (statistics are never computed before the split):
    1. Load default California Housing OR arbitrary user CSV (with automatic
       target inference and categorical feature encoding).
    2. Split indices 70 / 15 / 15.
    3. Median imputation from TRAIN split only -> impute all splits.
    4. 0.5th / 99.5th percentiles from TRAIN split only -> clip all splits.
    5. StandardScaler fit on clipped TRAIN split -> transform X across all splits.
    6. Target standardization (mean/std from y_train only) when --csv is used,
       with exact inverse transform metadata preserved for evaluation metrics.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CA_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]


def load_dataset(csv_path=None, target_col=None):
    """Return (X [N, D], y [N], meta_dict).

    - csv_path=None: loads sklearn California Housing (units of $100k).
    - csv_path=... : loads any tabular CSV. Uses `target_col` if specified,
      otherwise treats the last column as the continuous regression target.
      Automatically one-hot encodes non-numeric/categorical feature columns.
    """
    if csv_path is None:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        meta = {
            "dataset_name": "California Housing",
            "target_unit": "$100k",
            "scale_target": False,
            "feature_names": list(d.feature_names),
        }
        return d.data.astype(np.float64), d.target.astype(np.float64), meta

    import pandas as pd
    df = pd.read_csv(csv_path)

    # Support legacy Kaggle California Housing layout automatically if detected
    if target_col is None and "median_house_value" in df.columns and "total_rooms" in df.columns:
        hh = df["households"].replace(0, np.nan)
        X = np.column_stack([
            df["median_income"], df["housing_median_age"],
            df["total_rooms"] / hh, df["total_bedrooms"] / hh,
            df["population"], df["population"] / hh,
            df["latitude"], df["longitude"],
        ])
        y = df["median_house_value"].to_numpy(dtype=np.float64) / 1e5
        meta = {
            "dataset_name": Path_stem(csv_path),
            "target_unit": "$100k",
            "scale_target": False,
            "feature_names": CA_FEATURES,
        }
        return X.astype(np.float64), y, meta

    # General tabular CSV mode
    if target_col is None:
        target_col = df.columns[-1]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV columns: {list(df.columns)}")

    # Drop rows where target is missing or non-numeric
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    y = df[target_col].to_numpy(dtype=np.float64)
    X_df = df.drop(columns=[target_col])

    # Convert categorical/object columns via one-hot encoding (drop_first=True)
    cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dtype=np.float64)

    # Coerce any remaining columns to numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    # Drop columns that are 100% NaN
    X_df = X_df.dropna(axis=1, how="all")

    meta = {
        "dataset_name": Path_stem(csv_path),
        "target_unit": target_col,
        "scale_target": True,
        "feature_names": list(X_df.columns),
    }
    return X_df.to_numpy(dtype=np.float64), y, meta


def Path_stem(p):
    from pathlib import Path
    return Path(p).stem


def prepare(X, y, seed=42, scale_target=False):
    """Split, impute, clip, standardize. Returns a dict of float32 numpy arrays."""
    n = len(X)
    if n < 50:
        raise ValueError(f"Dataset too small ({n} rows) for 70/15/15 split.")

    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.30, random_state=seed, shuffle=True)
    va, te = train_test_split(rest, test_size=0.50, random_state=seed, shuffle=True)

    Xtr, Xva, Xte = (X[i].copy() for i in (tr, va, te))
    ytr_raw, yva_raw, yte_raw = y[tr].copy(), y[va].copy(), y[te].copy()

    # 1. Median imputation fitted strictly on TRAIN split
    train_median = np.nanmedian(Xtr, axis=0)
    # Fallback to 0.0 if a column happens to be all-NaN in train
    train_median = np.where(np.isnan(train_median), 0.0, train_median)
    for A in (Xtr, Xva, Xte):
        rows, cols = np.where(np.isnan(A))
        if len(rows) > 0:
            A[rows, cols] = train_median[cols]

    # 2. Outlier clipping: thresholds from the training split only
    lo, hi = np.percentile(Xtr, [0.5, 99.5], axis=0)
    Xtr, Xva, Xte = (np.clip(A, lo, hi) for A in (Xtr, Xva, Xte))

    # 3. Feature standardization: fit on the clipped training split only
    scaler = StandardScaler().fit(Xtr)
    # Guard against constant zero-variance columns
    scaler.scale_ = np.where(scaler.scale_ < 1e-12, 1.0, scaler.scale_)
    Xtr, Xva, Xte = (scaler.transform(A) for A in (Xtr, Xva, Xte))

    # 4. Target scaling (fitted strictly on y_train when arbitrary CSV is used)
    if scale_target:
        y_mean = float(np.mean(ytr_raw))
        y_std = float(np.std(ytr_raw))
        if y_std < 1e-12:
            y_std = 1.0
    else:
        y_mean, y_std = 0.0, 1.0

    ytr = (ytr_raw - y_mean) / y_std
    yva = (yva_raw - y_mean) / y_std
    yte = (yte_raw - y_mean) / y_std

    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    col = lambda a: f32(a.reshape(-1, 1))
    return {
        "X_train": f32(Xtr), "X_val": f32(Xva), "X_test": f32(Xte),
        "y_train": col(ytr), "y_val": col(yva), "y_test": col(yte),
        "y_test_orig": col(yte_raw),
        "y_mean": y_mean, "y_std": y_std,
        "clip_lo": lo, "clip_hi": hi,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "idx": {"train": tr, "val": va, "test": te},
    }