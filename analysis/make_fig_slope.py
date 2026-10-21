"""
Fig. 3: slope-based correction (20 GHz preamble data).

  a) 2D histogram of slope Delta t_BA = t_B - t_A vs t_A at one attenuation, with the calibration curve
     C_A(Delta t_BA) (median t_A in 4 ps slope bins, computed by the decoder: hist_data.corr1)
  b) slope histograms at three attenuations
  c) slope-corrected time tag histograms (t~_B, computed by the decoder: correction_data.corrected_hist2),
     with the +-25 ps slot boundaries of the 20 GHz demonstration

Times are in display coordinates (analysis/display.py); mu is the attenuator-calibrated mean photon number
from make_fig_decoding.py.

usage: uv run python -m analysis.make_fig_slope
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import orjson

from analysis.datasets import decode_path, get
from analysis.display import display_offset, one_photon_center

PANEL_A_DB = 26
HIST_DBS = (30, 26, 22)


def main():
    cfg = get("20")
    mu = {r["db"]: r["mu"] for r in json.loads(open(cfg["table"]).read())["rows"]}
    data = {db: orjson.loads(open(decode_path(cfg, db), "rb").read()) for db in HIST_DBS}
    colors = plt.cm.viridis(np.linspace(0, 1, 4))

    mpl.rcParams.update({"font.size": 10, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(11, 5))
    ax_a = fig.add_axes([0.07, 0.58, 0.38, 0.38])
    ax_b = fig.add_axes([0.07, 0.11, 0.38, 0.36])
    ax_c = fig.add_axes([0.56, 0.11, 0.42, 0.85])

    # a) slope vs t_A at one attenuation, with the calibration curve
    h = data[PANEL_A_DB]["hist_data"]
    off = display_offset(cfg, PANEL_A_DB)
    pts = np.asarray(h["counts"], float) + off
    dt = pts[:, 1] - pts[:, 0]
    d_off = off[1] - off[0]
    bins = np.asarray(h["bins"], float)
    ax_a.hist2d(dt, pts[:, 0], bins=(bins + d_off, bins + off[0]), norm=mpl.colors.LogNorm(), alpha=0.5,
                rasterized=True)
    ax_a.plot(np.asarray(h["slices"][:-1]) + d_off, np.asarray(h["corr1"]) + off[0], color="black")
    c1 = one_photon_center(cfg, PANEL_A_DB) + off  # 1-photon group: t_A = 0, Delta t_BA = waveform delay
    c1_dt = c1[1] - c1[0]
    ax_a.set_xlim(c1_dt - 180, c1_dt + 70)
    ax_a.set_ylim(-160, 40)
    ax_a.set_xlabel(r"$\Delta t_{BA}$ (ps)")
    ax_a.set_ylabel(r"$t_A$ (ps)", labelpad=0)
    ax_a.text(0.04, 0.93, rf"$\mu$ = {mu[PANEL_A_DB]:.2f}", transform=ax_a.transAxes, va="top", fontsize=13)

    # b) slope histograms (display offsets differ between files only by a common-mode shift, which does not
    #    change Delta t_BA, so the curves are directly comparable)
    for k, db in enumerate(HIST_DBS):
        p = np.asarray(data[db]["hist_data"]["counts"], float) + display_offset(cfg, db)
        hist, edges = np.histogram(p[:, 1] - p[:, 0], bins=bins + d_off)
        ax_b.plot(edges[:-1], hist, color=colors[k], alpha=0.7, label=rf"$\mu$ = {mu[db]:.2f}")
    ax_b.set_xlim(ax_a.get_xlim())
    ax_b.set_xlabel(r"$\Delta t_{BA}$ (ps)")
    ax_b.set_ylabel("counts")
    ax_b.legend(fancybox=False, framealpha=0.1)

    # c) slope-corrected time tags
    for k, db in enumerate(HIST_DBS):
        c = data[db]["correction_data"]
        ax_c.plot(np.asarray(c["corrected_bins"][:-1]), c["corrected_hist2"], color=colors[k], alpha=0.7,
                  label=rf"$\mu$ = {mu[db]:.2f}")
    half = cfg["slot_ps"] / 2
    lim = 78
    for s in (-1, 1):
        ax_c.axvline(s * half, color="red", alpha=0.5)
        ax_c.axvspan(s * half, s * lim, color="red", alpha=0.05)
    ax_c.set_xlim(-lim, lim)
    ax_c.set_xticks([-75, -50, -25, 0, 25, 50, 75])
    ax_c.set_xlabel(r"$\tilde{t}_B$ (ps)")
    ax_c.set_ylabel("counts")
    ax_c.legend(fancybox=False, framealpha=0.1)

    for ax, lab, y in [(ax_a, "a)", 1.12), (ax_b, "b)", 1.12), (ax_c, "c)", 1.05)]:
        ax.text(-0.12, y, lab, transform=ax.transAxes, fontsize=16, va="top")
    for x, y, n in [(c1_dt - 45, 12, "1"), (c1_dt - 110, -35, "2"), (c1_dt - 175, -80, "3+")]:
        ax_a.text(x, y, rf"$\mathbf{{|{n}\rangle}}$", fontsize=13)

    out = cfg["figs"] / "slope_cancellation_light"
    fig.savefig(f"{out}.pdf", dpi=300)
    fig.savefig(f"{out}.png", dpi=110)
    print(f"-> {out}.pdf")


if __name__ == "__main__":
    main()
