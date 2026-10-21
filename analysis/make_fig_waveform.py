"""
Fig. 2: rising edge of the detector pulse vs trigger level, and the two trigger levels used for t_A and t_B.

The scan (waveform/data/decent.npy) is a 2D histogram of time tags vs trigger voltage (5-60 mV, 1 ps
bins). Seven rows of the scan are realigned by a few bins (and rescaled by up to 15%) to repair a
time-tagger artefact that appeared at random trigger levels during the scan; the repaired rows are
listed in REPAIRED_ROWS and the figure caption says so.

Also writes waveform/data/tigger_level_time_difference.json: the 1-photon delay between the 8 mV and
50 mV crossings (centroid of each 1-photon peak), used as the t_B - t_A display offset (analysis/display.py).

usage: uv run python -m analysis.make_fig_waveform
"""

import json

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from analysis.datasets import REPO

DATA = REPO / "waveform" / "data"
FIGS = REPO / "waveform" / "figs"
TRIGGER_A_MV, TRIGGER_B_MV = 8, 50
V_RANGE_MV = (5, 60)
# (row index in the raw scan, columns, roll in bins, scale): repair of the time-tagger artefact
REPAIRED_ROWS = [(181, slice(0, 500), -5, 1.03), (182, slice(0, 500), -15, 1.05), (183, slice(0, 500), -20, 1.15),
                 (184, slice(0, 500), -13, 1.15), (185, slice(0, 500), -8, 1.05), (186, slice(0, 500), -3, 1.02),
                 (36, slice(None), -3, 0.7)]
CROP_ROWS, CROP_COLS = slice(15, None), slice(50, 1100)
TRIGGER_COLOR = "deeppink"


def load_scan():
    raw = np.load(DATA / "decent.npy")
    data = np.copy(raw).astype(float)
    for row, cols, roll, scale in REPAIRED_ROWS:
        data[row, cols] = np.roll(data[row, cols], roll) * scale
    scan = np.flip(data[CROP_ROWS, CROP_COLS], axis=0)  # rows now increase with trigger voltage
    volts = np.linspace(*V_RANGE_MV, scan.shape[0])
    return scan, volts


def row_at(volts, v):
    return int(np.interp(v, volts, np.arange(len(volts))))


def one_photon_delay(scan, volts):
    """centroid of the 1-photon peak at each trigger level (time windows as in the published analysis)"""
    t = np.arange(scan.shape[1])
    a = scan[row_at(volts, TRIGGER_A_MV), 188:400]
    b = scan[row_at(volts, TRIGGER_B_MV), 700:900]
    return np.average(t[700:900], weights=b) - np.average(t[188:400], weights=a)


def connect(fig, ax0, xy0, ax1, xy1, **kw):
    """line from data coords on ax0 to axes coords on ax1, in figure coordinates"""
    p0 = fig.transFigure.inverted().transform(ax0.transData.transform(xy0))
    p1 = fig.transFigure.inverted().transform(ax1.transAxes.transform(xy1))
    fig.add_artist(mpl.lines.Line2D([p0[0], p1[0]], [p0[1], p1[1]], transform=fig.transFigure, **kw))


def main():
    scan, volts = load_scan()
    delay = one_photon_delay(scan, volts)
    (DATA / "tigger_level_time_difference.json").write_text(json.dumps(delay))

    mpl.rcParams.update({"font.size": 10, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(8, 8))
    ax_top = fig.add_axes([0.1, 0.80, 0.85, 0.17])
    ax = fig.add_axes([0.1, 0.30, 0.85, 0.42])
    ax_bot = fig.add_axes([0.1, 0.07, 0.85, 0.16])

    viridis = plt.get_cmap("viridis")(np.linspace(0, 1, 256))
    t = np.arange(scan.shape[1])
    ax.imshow(scan, cmap=mcolors.LinearSegmentedColormap.from_list("v", viridis), aspect="auto",
              extent=[t[0], t[-1], *V_RANGE_MV], origin="lower")
    for v, (t0, t1), label, lx in [(TRIGGER_A_MV, (50, 300), f"Trigger A: {TRIGGER_A_MV} mV", 600),
                                   (TRIGGER_B_MV, (450, 900), f"Trigger B: {TRIGGER_B_MV} mV", 150)]:
        ax.axhline(v, color=TRIGGER_COLOR, alpha=0.3)
        ax.plot([t0, t1], [v, v], color=TRIGGER_COLOR)
        for tt in (t0, t1):
            ax.plot([tt, tt], [v - 0.8, v + 0.8], color=TRIGGER_COLOR)
        ax.text(lx, v + 2, label, ha="center", va="center", color=TRIGGER_COLOR, fontsize=13, alpha=0.6)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Voltage (mV)")

    for a, v, (t0, t1), corners in [(ax_top, TRIGGER_B_MV, (450, 900), ((0, 0), (1, 0))),
                                    (ax_bot, TRIGGER_A_MV, (50, 300), ((0, 1), (1, 1)))]:
        a.plot(t[t0:t1], scan[row_at(volts, v), t0:t1], color="k")
        a.set_xlabel("Time (ps)")
        a.set_ylabel("Counts")
        connect(fig, ax, (t0, v), a, corners[0], color="lightgray", alpha=0.7)
        connect(fig, ax, (t1, v), a, corners[1], color="lightgray", alpha=0.7)
    for x, y, n in [(780, 300, "1"), (620, 200, "2"), (555, 110, "3")]:
        ax_top.text(x, y, rf"$\mathbf{{|{n}\rangle}}$", fontsize=14)

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "waveform_light.pdf", dpi=300)
    fig.savefig(FIGS / "waveform_light.png", dpi=110)
    print(f"1-photon trigger delay {delay:.2f} ps -> {FIGS / 'waveform_light.pdf'}")


if __name__ == "__main__":
    main()
