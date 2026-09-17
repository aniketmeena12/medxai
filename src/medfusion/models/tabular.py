"""B1: metadata-only baseline (gradient boosting on the encoded metadata)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from medfusion.data.metadata import MetadataSchema


def metadata_matrix(schema: MetadataSchema, df) -> tuple[np.ndarray, list[bool]]:
    """Categorical codes (MISSING as its own code) + continuous values (NaN when missing)."""
    cat, cont, mask = schema.encode(df)
    cont = np.where(mask, cont, np.nan)
    x = np.concatenate([cat.astype(np.float64), cont.astype(np.float64)], axis=1)
    is_cat = [True] * cat.shape[1] + [False] * cont.shape[1]
    return x, is_cat


def fit_b1(schema: MetadataSchema, train_df, y_train, seed: int = 0):
    x, is_cat = metadata_matrix(schema, train_df)
    clf = HistGradientBoostingClassifier(
        categorical_features=np.array(is_cat), class_weight="balanced", random_state=seed,
        early_stopping=True, validation_fraction=0.15,
    )
    return clf.fit(x, y_train)
