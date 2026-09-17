"""P4: counterfactual metadata test (contribution N2).

Keep the image fixed, change one metadata field, and measure how far the image attribution moves.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from medfusion.data.metadata import MetadataSchema


def map_shift(before: np.ndarray, after: np.ndarray) -> float:
    """1 - Spearman rank correlation between two maps (0 = identical ranking)."""
    b, a = np.asarray(before).ravel(), np.asarray(after).ravel()
    if np.ptp(b) == 0 or np.ptp(a) == 0:
        return 0.0 if np.allclose(a, b) else 1.0
    return float(1 - spearmanr(b, a).statistic)


def perturb_field(
    schema: MetadataSchema,
    field: str,
    cat: torch.Tensor,
    cont: torch.Tensor,
    cont_mask: torch.Tensor,
    generator: torch.Generator | None = None,
    cont_shift_sd: float = 1.0,
):
    """Return copies with ``field`` changed for every row.

    Categorical: a different, non-missing category drawn uniformly (rows whose value is MISSING
    get a random observed category). Continuous: shifted by ``cont_shift_sd`` standard deviations
    (sign chosen at random) and marked present.
    """
    cat, cont, cont_mask = cat.clone(), cont.clone(), cont_mask.clone()
    if field in schema.categorical:
        j = schema.categorical.index(field)
        k = len(schema.vocabs[field])
        if k < 2:
            raise ValueError(f"field {field!r} has fewer than 2 categories")
        current = cat[:, j]
        offset = torch.randint(1, k, current.shape, generator=generator, device=current.device)
        new = torch.where(current >= k,
                          torch.randint(0, k, current.shape, generator=generator,
                                        device=current.device),
                          (current + offset) % k)
        cat[:, j] = new
    elif field in schema.continuous:
        j = schema.continuous.index(field)
        sign = torch.randint(0, 2, (cont.shape[0],), generator=generator,
                             device=cont.device) * 2 - 1
        cont[:, j] = cont[:, j] + sign * cont_shift_sd
        cont_mask[:, j] = True
    else:
        raise KeyError(f"unknown field {field!r}")
    return cat, cont, cont_mask
