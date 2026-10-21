"""
Display coordinates for (t_A, t_B) panels (Figs. 3-6, S1 and the 10.75 GHz supplemental figures).

Detections and preamble points are referenced to their PPM slot, but each attenuation setting carries an
arbitrary time offset (the decoder aligns every file to its own slope-corrected arrival time).
pnr_refit.py fits that offset as a common-mode shift s of the photon-number model, so removing it puts
the 1-photon group of every dataset in the same place: t_A = 0. t_B is then offset by a constant (the
same for every attenuation of a dataset) so that the 1-photon t_B - t_A equals the delay between the two
trigger crossings measured from the detector waveform (Fig. 2), i.e. the fixed cable delay between the
two time-tagger channels is removed. Without a waveform delay for a dataset only the common-mode shift is
applied and t_B - t_A keeps the cable delay.

    from analysis.display import display_offset
    xy_display = xy_slot_referenced + display_offset(cfg, db)
"""

import json
from functools import lru_cache

import numpy as np

from analysis.datasets import REPO
from analysis.pnr_refit import load_groups

WAVEFORM_DELAY_FILE = REPO / "waveform" / "data" / "tigger_level_time_difference.json"


@lru_cache
def waveform_delay_ps() -> float:
    """1-photon delay between the 8 mV and 50 mV trigger crossings, from the waveform scan (Fig. 2)."""
    return float(json.loads(WAVEFORM_DELAY_FILE.read_text()))


def one_photon_center(cfg: dict, db: float) -> np.ndarray:
    """(t_A, t_B) of the 1-photon group in slot-referenced coordinates, from the fitted model shift."""
    groups = load_groups(cfg["model"])
    c1 = np.average(groups[0]["means"], weights=groups[0]["weights"], axis=0)
    shift = json.loads(open(cfg["refit"]).read())[f"{float(db):.1f}"]["shift"]
    return c1 + np.asarray(shift)


def display_offset(cfg: dict, db: float) -> np.ndarray:
    """vector to add to slot-referenced (t_A, t_B) to get display coordinates"""
    return offset_for_center(cfg, one_photon_center(cfg, db))


def offset_for_center(cfg: dict, c1) -> np.ndarray:
    """display offset given the slot-referenced 1-photon centre (for a model shift fit elsewhere)"""
    c1 = np.asarray(c1)
    if cfg.get("display_delay") == "waveform":
        return np.array([-c1[0], waveform_delay_ps() - c1[1]])
    return np.array([-c1[0], -c1[0]])
