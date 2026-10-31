"""Fixed statistical parameters and linear quantile interpolation."""
import numpy as np

SEED = 20260927

BOOTSTRAP_REPLICATES = 20_000

def formal_parameters(replicates: int, seed: int) -> bool:
    """G6c: only the pre-registered 20,000 replicates and seed may be formal."""
    return type(replicates) is int and type(seed) is int and (
        replicates == BOOTSTRAP_REPLICATES and seed == SEED)

def linear_quantile(sorted_values: np.ndarray, p: float) -> float:
    """Endpoint interpolation fixed in spec 0051: zero-based p*(N-1)."""
    values = np.asarray(sorted_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not 0 <= p <= 1:
        raise ValueError("nonempty sorted vector and p in [0,1] required")
    position = p * (len(values) - 1)
    lower = int(np.floor(position))
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[min(lower + 1, len(values) - 1)] * weight)
