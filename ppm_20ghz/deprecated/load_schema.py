from pydantic import BaseModel, validator
import numpy as np
from enum import Enum
from dataclasses import dataclass, fields, field, _MISSING_TYPE
import numpy.typing as npt


class Result(Enum):
    CORRECT = "A"
    INCORRECT = "B"
    MISSING = "D"
    INCORRECT_EXTRA = "C"
    DEADTIME_ERROR = "E"


class PhotonNumberMeasurement(BaseModel):
    guess: int
    probability: float
    correction: int
    measured: int | None = None


class Event(BaseModel):
    result: Result
    measured: int = -1
    gaussian_measured: int = -1

    true: int = -1
    tag: float = -1.0
    tag_x: float = -1.0
    tag_y: float = -1.0
    pnr_best: PhotonNumberMeasurement | None = None
    pnr_2nd_best: PhotonNumberMeasurement | None = None


class GMData(BaseModel):
    """stores data for gaussian mixture that models detector response for a given mean photon rate.
    Contains gaussians that model multiple photon number groupings
    """

    num_components: int
    log_likelihood: float
    covariances: list  # [npt.NDArray[np.float64]]
    means: list  # [npt.NDArray[np.float64]]
    weights: list  # [npt.NDArray[np.float64]]

    @validator("covariances", "means", "weights")
    def to_numpy_array(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

    class Config:
        arbitrary_types_allowed = True


class GMDataPhotonCommunity(BaseModel):
    """stores data for gaussian mixture that models ONE PNR grouping"""

    photon_num_components: int
    photon_covariances: list[npt.NDArray[np.float64]]
    photon_means: list[npt.NDArray[np.float64]]
    photon_weights: list[npt.NDArray[np.float64]]

    @validator("photon_covariances", "photon_means", "photon_weights")
    def to_numpy_array(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

    class Config:
        arbitrary_types_allowed = True


class GMTotalData(BaseModel):
    gm_list: list[GMData]
    counts: list

    @validator("counts")
    def to_numpy_array(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v


class CorrectionData(BaseModel):
    corrected_hist1: list
    corrected_hist2: list
    corrected_bins: list
    uncorrected_bins: list
    uncorrected_hist1: list
    uncorrected_hist2: list

    @validator(
        "corrected_hist1",
        "corrected_hist2",
        "corrected_bins",
        "uncorrected_bins",
        "uncorrected_hist1",
        "uncorrected_hist2",
    )
    def to_numpy_array(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

    class Config:
        arbitrary_types_allowed = True


class PNRHistCorrectionData(BaseModel):
    counts: list
    corr1: list
    corr2: list
    bins: list
    slices: list

    @validator("counts", "corr1", "corr2", "bins", "slices")
    def to_numpy_array(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

    class Config:
        arbitrary_types_allowed = True


class Decode(BaseModel):
    results: list[list[Event]]
    gm_data: GMTotalData
    hist_data: PNRHistCorrectionData
    correction_data: CorrectionData

    # @validator('results')
    # def cnv(cls, v):
    #     return v[0]
