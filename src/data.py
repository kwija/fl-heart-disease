"""
Data loading and preprocessing for the Heart Disease UCI dataset.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from .config import BATCH_SIZE


def load_and_prepare_data(seed=42):
    """
    Load UCI Heart Disease dataset with distribution logging.

    Returns a dict with:
      - X_train, y_train, X_val, y_val, X_test, y_test (tensors)
      - sex_train, sex_val, sex_test (numpy arrays)
      - n_train, n_val, n_test (ints)
      - input_dim (int)
      - pos_rate_train, pos_rate_val, pos_rate_test (floats)
    """
    from ucimlrepo import fetch_ucirepo

    np.random.seed(seed)
    hd = fetch_ucirepo(id=45)
    data = pd.concat([hd.data.features, hd.data.targets], axis=1).dropna()
    data["target"] = (data["num"] > 0).astype(int)
    data = data.drop("num", axis=1)
    sex_original = data["sex"].values.copy()
    X = data.drop("target", axis=1)
    y = data["target"]
    input_dim = X.shape[1]

    X_tv, X_test, y_tv, y_test, sex_tv, sex_test = train_test_split(
        X, y, sex_original, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val, sex_train, sex_val = train_test_split(
        X_tv, y_tv, sex_tv, test_size=0.25, random_state=seed, stratify=y_tv
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return {
        "X_train": torch.FloatTensor(X_train_s),
        "y_train": torch.LongTensor(y_train.values),
        "X_val": torch.FloatTensor(X_val_s),
        "y_val": torch.LongTensor(y_val.values),
        "X_test": torch.FloatTensor(X_test_s),
        "y_test": torch.LongTensor(y_test.values),
        "sex_train": sex_train,
        "sex_val": sex_val,
        "sex_test": sex_test,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "input_dim": input_dim,
        "pos_rate_train": y_train.mean(),
        "pos_rate_val": y_val.mean(),
        "pos_rate_test": y_test.mean(),
    }


def make_train_loader(X_c, y_c, batch_size=BATCH_SIZE):
    """
    Create a DataLoader with BatchNorm safety guards.

    Returns (loader, effective_n):
      - Skips clients with n < 2 (BatchNorm impossible in train mode).
      - Uses drop_last if the last batch would have size 1.
    """
    n = int(len(X_c))
    if n < 2:
        return None, 0

    bs = min(int(batch_size), n)
    if bs < 2:
        return None, 0

    drop_last = n % bs == 1
    eff_n = n - 1 if drop_last else n

    loader = DataLoader(
        TensorDataset(X_c, y_c),
        batch_size=bs,
        shuffle=True,
        drop_last=drop_last,
    )
    return loader, eff_n
