import numpy as np


LANDMARK_DIMENSIONS = [2, 3, 4, 5, 6, 7]


def discretize(
    observation,
    bins=5,
    low=-2,
    high=2,
    dimensions=LANDMARK_DIMENSIONS,
):
    """
    Converts a continuous observation into a discrete state.

    Parameters
    ----------
    observation : np.ndarray
        Continuous observation from the environment.

    bins : int
        Number of intervals per dimension.

    low : float
        Minimum expected value.

    high : float
        Maximum expected value.

    dimensions : list
        Observation indexes used to create the state.

    Returns
    -------
    tuple
        Discrete state representation.
    """

    selected_values = observation[dimensions]

    clipped_values = np.clip(
        selected_values,
        low,
        high
    )

    discrete_values = np.floor(
        (clipped_values - low)
        /
        (high - low)
        *
        bins
    ).astype(int)

    discrete_values = np.minimum(
        discrete_values,
        bins - 1
    )

    return tuple(discrete_values)