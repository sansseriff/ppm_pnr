"""Per-dataset settings shared by the analysis scripts. All paths are relative to the repository root.

    from analysis.datasets import get
    cfg = get("20")    # or "10"
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DATASETS = {
    "20": dict(
        name="ppm_20ghz",
        tag="20GHz",
        prefix="decode_20GHz",
        dbs=list(range(16, 50, 2)),
        laser_time_ps=50.0,  # the 20 GHz decode files carry no set_data, so the slot width is given here
        slot_ps=50.0,
        bits=11,  # M = 2048 slots per frame
        # GMM used by the SLOPE vs GMM comparison: gm_list[10] = 25 components (as in the original Fig. 5)
        gm_idx=10,
        # Fig. 5 panels
        image_dbs=(36, 28, 24),
        scatter_dbs=(28, 24),
        flagged_dbs=(),
        ylim=(1e-2, 1.15),
        # photon-number model: built from the 24 dB preamble (ppm_20ghz/deprecated/bi_gauss_community_detection.ipynb, rebuilt by build_pnr_model.py);
        # Fig. 6 shows the 26 dB data
        model_db=24,
        group_db=26,
    ),
    "10": dict(
        name="ppm_10ghz",
        tag="10GHz",
        prefix="decode_10GHz",
        dbs=list(range(18, 46, 2)),
        laser_time_ps=None,  # None: taken from the decode files (set_data.laser_time = 93.02 ps)
        slot_ps=1e12 / 10.75e9,  # slot width used by the photon-number scripts
        bits=10,  # M = 1024
        gm_idx=-1,  # last stored fit = 23 components
        image_dbs=(40, 32, 28),
        scatter_dbs=(32, 28),
        flagged_dbs=(),  # 42 dB is the re-decoded file with the corrected window (see the supplement)
        ylim=(1e-4, 1.15),
        model_db=28,  # analysis/build_pnr_model.py --dataset 10 (35-component refit of the 28 dB preamble)
        group_db=30,
    ),
}


def get(key: str) -> dict:
    cfg = dict(DATASETS[str(key)])
    root = REPO / cfg["name"]
    cfg.update(
        key=str(key),
        root=root,
        inter=root / "inter",
        figs=root / "figs",
        model=root / "inter" / "static_sim.json",
        refit=root / "inter" / "pnr_refit.json",
        fair=root / "inter" / "fair_comparison.json",
        table=root / "figs" / f"ppm_decoding_{cfg['tag']}_table.json",
    )
    return cfg


def decode_path(cfg: dict, db: float) -> Path:
    return cfg["inter"] / f"{cfg['prefix']}{float(db):.1f}.json"
