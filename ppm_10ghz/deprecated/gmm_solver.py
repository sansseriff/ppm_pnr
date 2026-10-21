import numpy as np
from numba import njit
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, validator

from load_schema import GMData, Result, Event


@njit
def gaussian_2d(x, y, cov, pos=(0, 0)):
    """Generate a 2D Gaussian distribution with a given covariance matrix and position."""
    # Unpack the position tuple
    x0, y0 = pos

    # Shift the coordinates by x0 and y0
    x = x - x0
    y = y - y0

    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)

    # Calculate the determinant of the covariance matrix
    det_cov = np.linalg.det(cov)

    # Calculate the exponent term of the Gaussian function
    exponent = -0.5 * (
        inv_cov[0, 0] * x**2
        + (inv_cov[0, 1] + inv_cov[1, 0]) * x * y
        + inv_cov[1, 1] * y**2
    )

    # Calculate the normalization constant of the Gaussian function
    norm_const = 1 / (2 * np.pi * np.sqrt(det_cov))

    # Calculate the Gaussian function values
    gauss = norm_const * np.exp(exponent)

    return gauss

@njit
def find_gm_prob_for_offset(
    xy,
    offset_idx,
    gm_means,
    gm_covar,
    gm_weights,
    bin_width=50,
):
    """given the gaussian mixture model for the distribution of counts
    at a given offset index, find the probability of the count originating
    from the time slot with this offset. The offset index is a number of
    bins, (probably 50 ps wide)
    """
    x = xy[0]
    y = xy[1]
    z = 0

    offset = offset_idx * bin_width

    for pos, covar, w in zip(gm_means, gm_covar, gm_weights):
        z = z + gaussian_2d(x, y, covar, [pos[0] + offset, pos[1] + offset]) * w

    return z


def correction_from_gaussian_model(estimate, dual_tag, gm_data: GMData, laser_time, offset = 0.0):
    """_summary_

    Args:
        estimate (_type_): an index 0 to 2048 or 1024
        dual_tag (_type_): _description_
        gm_data (GMData): _description_
        laser_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    offs = offset

    prob_4 = find_gm_prob_for_offset(
        dual_tag,
        estimate - 1 + offs,
        gm_data.means,
        gm_data.covariances,
        gm_data.weights,
        bin_width=laser_time,
    )
    prob_5 = find_gm_prob_for_offset(
        dual_tag,
        estimate - 0 + offs,
        gm_data.means,
        gm_data.covariances,
        gm_data.weights,
        bin_width=laser_time,
    )
    prob_6 = find_gm_prob_for_offset(
        dual_tag,
        estimate + 1 + offs,
        gm_data.means,
        gm_data.covariances,
        gm_data.weights,
        bin_width=laser_time,
    )
    

    # print("prob 1: ", prob_1, " prob 2: ", prob_2, " prob 3: ", prob_3)
    largest_idx = np.argmax(
        [prob_4, prob_5, prob_6]
    )

    return int(largest_idx - 1)
    # return 0