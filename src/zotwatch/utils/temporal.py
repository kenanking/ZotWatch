"""Temporal weight computation utilities.

Implements exponential decay weighting for time-sensitive relevance scoring.
Papers that were added more recently have higher weights, reflecting the user's
current research interests.
"""

import math
from datetime import datetime

from zotwatch.utils.datetime import utc_now


def compute_temporal_weight(
    date_added: datetime | None,
    halflife_days: float = 180.0,
    min_weight: float = 0.05,
    reference_time: datetime | None = None,
) -> float:
    """Compute exponential decay weight for a paper.

    Formula: w = exp(-ln(2) / T_half * (t_now - t_i))

    At t_now - t_i = 0 (just added): w = 1.0
    At t_now - t_i = T_half: w = 0.5
    At t_now - t_i = 2 * T_half: w = 0.25

    Args:
        date_added: When item was added to library (timezone-aware datetime).
        halflife_days: Half-life period in days. Default 180 (6 months).
        min_weight: Minimum weight floor to prevent zero weights. Default 0.05.
        reference_time: Reference time for age calculation (defaults to now).

    Returns:
        Weight in [min_weight, 1.0].
    """
    if date_added is None:
        return 1.0  # Treat missing date as recently added

    ref_time = reference_time or utc_now()

    # Compute age in days
    age_delta = ref_time - date_added
    age_days = age_delta.total_seconds() / 86400  # Convert to days

    if age_days <= 0:
        return 1.0  # Future date or same time

    # Exponential decay: w = exp(-ln(2) / T_half * age)
    decay_rate = math.log(2) / halflife_days
    weight = math.exp(-decay_rate * age_days)

    return max(min_weight, weight)


def compute_batch_weights(
    items: list,
    halflife_days: float = 180.0,
    min_weight: float = 0.05,
) -> list[float]:
    """Compute temporal weights for a batch of items.

    Args:
        items: List of objects with date_added attribute.
        halflife_days: Half-life period in days.
        min_weight: Minimum weight floor.

    Returns:
        List of weights corresponding to each item.
    """
    ref_time = utc_now()
    return [
        compute_temporal_weight(
            getattr(item, "date_added", None),
            halflife_days,
            min_weight,
            ref_time,
        )
        for item in items
    ]


def compute_item_age_days(
    date_added: datetime | None,
    reference_time: datetime | None = None,
) -> float:
    """Compute age of an item in days.

    Args:
        date_added: When item was added to library.
        reference_time: Reference time (defaults to now).

    Returns:
        Age in days (float). Returns 0 if date_added is None.
    """
    if date_added is None:
        return 0.0

    ref_time = reference_time or utc_now()
    age_delta = ref_time - date_added
    return max(0.0, age_delta.total_seconds() / 86400)


__all__ = [
    "compute_temporal_weight",
    "compute_batch_weights",
    "compute_item_age_days",
]
