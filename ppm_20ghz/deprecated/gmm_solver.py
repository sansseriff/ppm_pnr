import numpy as np
from numba import njit
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, validator

from load_schema import GMData, Result, Event
from util import PhotonGMData, GMDataPhotonCommunity

from load_schema import PhotonNumberMeasurement


@njit
def gaussian_2d(x: float, y: float, cov, pos=(0, 0)) -> float:
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
    exponent: float = -0.5 * (
        inv_cov[0, 0] * x**2
        + (inv_cov[0, 1] + inv_cov[1, 0]) * x * y
        + inv_cov[1, 1] * y**2
    )

    # Calculate the normalization constant of the Gaussian function
    norm_const = 1 / (2 * np.pi * np.sqrt(det_cov))

    # Calculate the Gaussian function values
    gauss: float = norm_const * np.exp(exponent)

    return gauss


@njit
def find_gm_prob_for_offset(
    xy: tuple[float, float],
    offset_idx: float,
    gm_means,
    gm_covar,
    gm_weights,
    bin_width: float | int = 50,
) -> float:
    """given the gaussian mixture model for the distribution of counts
    at a given offset index, find the probability of the count originating
    from the time slot with this offset. The offset index is a number of
    bins, (probably 50 ps wide)
    """
    x = xy[0]
    y = xy[1]
    z: float = 0

    offset = offset_idx * bin_width

    for pos, covar, w in zip(gm_means, gm_covar, gm_weights):
        z = z + gaussian_2d(x, y, covar, [pos[0] + offset, pos[1] + offset]) * w

    return z


# def arrival_correction_and_pnr_from_gaussian_model(estimate, dual_tag,


def photon_data_from_gaussian_model(
    ls: list[PhotonGMData],
    dual_tag: tuple[float, float],
    estimate_idx: float,
    laser_time: float | int = 50,
):
    """
    estimate_idx is in index 'space' but it can be non-integer
    """
    photon_probabilities: list[float] = []
    for community in ls:
        community = community.gm_photon
        photon_probabilities.append(
            find_gm_prob_for_offset(
                dual_tag,
                estimate_idx,
                community.photon_means,
                community.photon_covariances,
                community.photon_weights,
                bin_width=laser_time,
            )
        )

    photon_numbers = np.arange(1, len(photon_probabilities) + 1).tolist()

    return photon_numbers, photon_probabilities


def pnr_correction_from_gaussian_model(
    estimate: int,
    dual_tag: tuple[float, float],
    gm_data: list[PhotonGMData],
    laser_time: float | int = 50,
    offset: float = 0.0,
) -> tuple[PhotonNumberMeasurement, PhotonNumberMeasurement]:
    assert isinstance(gm_data, list)
    assert isinstance(gm_data[0], PhotonGMData)
    ofs = [-1, 0, 1]

    full_photon_numbers = []
    full_photon_probs = []
    full_arrival_time_idx_list = []
    idx = 0
    for of in ofs:
        arrival_time_idx = idx
        photon_numbers, photon_probs = photon_data_from_gaussian_model(
            gm_data, dual_tag, estimate + of + offset, laser_time
        )
        arrival_time_idx_list = [arrival_time_idx] * len(photon_probs)
        full_photon_numbers.extend(photon_numbers)
        full_photon_probs.extend(photon_probs)
        full_arrival_time_idx_list.extend(arrival_time_idx_list)
        idx += 1

    sorted_args = np.argsort(full_photon_probs)
    best_photon_number = full_photon_numbers[sorted_args[-1]]
    best_2nd_photon_number = full_photon_numbers[sorted_args[-2]]
    best_time_idx = ofs[
        full_arrival_time_idx_list[sorted_args[-1]]
    ]  # ofs is -1 to 1, arrival_time_idx_list is 0 to 2
    best_2nd_time_idx = ofs[full_arrival_time_idx_list[sorted_args[-2]]]
    best_prob = full_photon_probs[sorted_args[-1]]
    best_2nd_prob = full_photon_probs[sorted_args[-2]]

    pnr_best = PhotonNumberMeasurement(
        guess=best_photon_number,
        probability=best_prob,
        correction=best_time_idx,
        measured=None,
    )
    pnr_2nd_best = PhotonNumberMeasurement(
        guess=best_2nd_photon_number,
        probability=best_2nd_prob,
        correction=best_2nd_time_idx,
        measured=None,
    )

    return pnr_best, pnr_2nd_best


def correction_from_gaussian_model(
    estimate: int,
    dual_tag: tuple[float, float],
    gm_data: GMData,
    laser_time: float | int = 50,
    offset: float = 0.0,
):
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
        bin_width=50,
    )
    prob_5 = find_gm_prob_for_offset(
        dual_tag,
        estimate - 0 + offs,
        gm_data.means,
        gm_data.covariances,
        gm_data.weights,
        bin_width=50,
    )
    prob_6 = find_gm_prob_for_offset(
        dual_tag,
        estimate + 1 + offs,
        gm_data.means,
        gm_data.covariances,
        gm_data.weights,
        bin_width=50,
    )

    # print("prob 1: ", prob_1, " prob 2: ", prob_2, " prob 3: ", prob_3)
    largest_idx = np.argmax([prob_4, prob_5, prob_6])

    return int(largest_idx - 1)
    # return 0
