"""Typed loaders (pydantic v2) for the decode files inter/decode_{10,20}GHz{dB}.0.json.

The decoder writes plain lists; the array-valued fields below are converted to numpy arrays on load
(`NDArray`), and back to lists when a model is serialised with model_dump_json() / model_dump(mode="json").

    import orjson
    from analysis.load_schema import Decode
    decode = Decode(**orjson.loads(open("ppm_20ghz/inter/decode_20GHz26.0.json", "rb").read()))

Migrated from the pydantic 1.9 versions in the original ppm_20ghz / ppm_10ghz repositories:
@validator(...) -> NDArray annotated type (BeforeValidator), class Config -> model_config.
"""

from enum import Enum
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer


def _to_array(v):
    return np.asarray(v) if isinstance(v, (list, tuple)) else v


def _to_list(v):
    return v.tolist() if isinstance(v, np.ndarray) else v


# a list in the JSON file, a numpy array in python
NDArray = Annotated[Any, BeforeValidator(_to_array), PlainSerializer(_to_list, when_used="json")]


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
    """One gaussian mixture fit (sklearn GaussianMixture) of the preamble detector response."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_components: int
    log_likelihood: float
    covariances: NDArray
    means: NDArray
    weights: NDArray


class GMDataPhotonCommunity(BaseModel):
    """Gaussian mixture that models ONE photon-number group."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    photon_num_components: int
    photon_covariances: NDArray
    photon_means: NDArray
    photon_weights: NDArray


class GMTotalData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gm_list: list[GMData]
    counts: NDArray


class CorrectionData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    corrected_hist1: NDArray
    corrected_hist2: NDArray
    corrected_bins: NDArray
    uncorrected_bins: NDArray
    uncorrected_hist1: NDArray
    uncorrected_hist2: NDArray


class PNRHistCorrectionData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    counts: NDArray
    corr1: NDArray
    corr2: NDArray
    bins: NDArray
    slices: NDArray


class Decode(BaseModel):
    results: list[list[Event]]
    gm_data: GMTotalData
    hist_data: PNRHistCorrectionData
    correction_data: CorrectionData
    set_data: dict = {}  # sequence parameters; present in the 10 GHz files only
