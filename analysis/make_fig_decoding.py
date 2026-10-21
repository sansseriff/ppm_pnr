"""
PPM decoding performance figure (manuscript Fig. 5 for 20 GHz, supplemental for 10 GHz), from the
symmetric, cross-validated SLOPE vs GMM comparison in fair_comparison.py.

  a) fraction of frames decoded correctly / incorrectly / missing vs mean photon number mu, both methods,
     with the relative error-rate reduction annotated where the paired McNemar test gives p < 0.01
  b) the transmitted image as decoded (GMM method, one transmission) at three mu
  c, d) (t_A, t_B) of detections referenced to their true slot, coloured by which method decoded them

mu is the mean detected photon number per pulse, fit as mu = A * 10^(-dB/10) to -ln(vacuum fraction)
over the attenuations where the vacuum fraction is a reliable estimator.

usage: uv run python -m analysis.make_fig_decoding --dataset 20   (or 10)   [--recompute]
"""

import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from analysis.datasets import get
from analysis.display import display_offset, one_photon_center
from analysis.fair_comparison import cached_analysis

DEAD_TIME_PS = 250e3  # from the ground-truth params (minimum_dead_time: 250 ns)
MU_FIT_RANGE = (0.03, 3.0)  # vacuum-fraction mu used for the attenuator-law fit
P_SIGNIFICANT = 0.01

COLORS = dict(
    correct="#7a3ff2",
    incorrect="#e8302a",
    missing="#222222",
    both="#7a3ff2",
    fixed="#1fbf1f",
    broken="#ff2b2b",
    copies="#525252",
)
MU_COLORS = ["#2e9e5b", "#1f6fd1", "#f08a24"]


def fit_mu(dbs, mu_vac):
    dbs, mu_vac = np.asarray(dbs, float), np.asarray(mu_vac)
    sel = (mu_vac > MU_FIT_RANGE[0]) & (mu_vac < MU_FIT_RANGE[1])
    lin = 10 ** (-dbs / 10)
    A = np.exp(np.mean(np.log(mu_vac[sel] / lin[sel])))  # least squares in log space
    return A * lin, A, sel


def canonical_truth(cfg):
    """true symbol of every frame index. Padding events appended by decode_ppm carry true=-1, so
    take the value seen at that frame index in any stream of any file."""
    truth = None
    for db in cfg["dbs"]:
        _, ev = cached_analysis(cfg, db)
        if truth is None:
            n = int(ev["frame"].max()) + 1
            truth = np.full(n + 64, -1)
        ok = ev["true"] >= 0
        truth[ev["frame"][ok]] = ev["true"][ok]
    return truth[: np.max(np.nonzero(truth >= 0)) + 1]


def decoded_image(symbols, bits):
    """symbols (frames,) -> 64x64 RGB uint8. The image bytes were sent MSB-first, `bits` per frame."""
    s = np.where(symbols < 0, 0, symbols).astype(np.int64)
    b = ((s[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    return np.packbits(b[: 64 * 64 * 3 * 8]).reshape(64, 64, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["10", "20"], required=True)
    ap.add_argument("--recompute", action="store_true", help="ignore cached fair_events_*.npz")
    args = ap.parse_args()
    cfg = get(args.dataset)
    cfg["out"] = str(cfg["figs"] / f"ppm_decoding_{cfg['tag']}")

    summaries, events = {}, {}
    for db in cfg["dbs"]:
        summaries[db], events[db] = cached_analysis(cfg, db, recompute=args.recompute)
        print(f"{db} dB loaded", flush=True)
    dbs = np.array(cfg["dbs"])
    T = summaries[dbs[0]]["laser_time_ps"]
    mu_vac = np.array([summaries[d]["mu_raw"] for d in dbs])
    mu, A, mu_sel = fit_mu(dbs, mu_vac)
    frame_ps = (2 ** cfg["bits"]) * T + DEAD_TIME_PS
    raw_rate = cfg["bits"] / frame_ps * 1e6  # Mbit/s (bits per ps * 1e6)

    n_tot = np.array([summaries[d]["n_total"] for d in dbs])
    n_mis = np.array([summaries[d]["n_missing"] for d in dbs])
    n_det = np.array([summaries[d]["n_detected"] for d in dbs])
    e_g = np.array([summaries[d]["gmm_errors"] for d in dbs])
    e_s = np.array([summaries[d]["slope_errors"] for d in dbs])
    pvals = np.array([summaries[d]["mcnemar_p"] for d in dbs])
    frac = lambda k: k / n_tot
    ferr = lambda k: np.sqrt(np.maximum(k, 1)) / n_tot

    mpl.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "font.family": "sans-serif",
                         "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(13, 6.5))
    ax_a = fig.add_axes([0.07, 0.42, 0.41, 0.55])
    ax_b = [fig.add_axes([0.07 + i * 0.145, 0.07, 0.12, 0.24]) for i in range(3)]
    ax_c = fig.add_axes([0.54, 0.08, 0.2, 0.88])
    ax_d = fig.add_axes([0.79, 0.08, 0.2, 0.88])

    # ---- a) decoding probabilities
    c = COLORS
    mu_all = mu.copy()
    full = dict(mu=mu, n_tot=n_tot, n_det=n_det, e_g=e_g, e_s=e_s, pvals=pvals)
    flag = np.isin(dbs, cfg["flagged_dbs"])
    if flag.any():
        # shown unconnected with open markers; excluded from the curves
        for k, col in [(n_det - e_g, c["correct"]), (e_g, c["incorrect"]), (e_s, c["incorrect"])]:
            ax_a.plot(mu[flag], frac(k)[flag], "o", mfc="none", color=col, ms=5)
        ax_a.annotate("frame-phase\nfault (excluded)", (mu[flag][0], frac(e_g)[flag][0]), xytext=(6, 0),
                      textcoords="offset points", fontsize=7, color=c["incorrect"], va="center")
        keep = ~flag
        mu, n_tot, n_mis, n_det, e_g, e_s, pvals = (a[keep] for a in (mu, n_tot, n_mis, n_det, e_g, e_s, pvals))
    ax_a.plot(mu, frac(n_det - e_g), "-o", color=c["correct"], ms=4, label="correct GMM")
    ax_a.plot(mu, frac(n_det - e_s), ":o", color=c["correct"], ms=2.5, label="correct SLOPE")
    ax_a.errorbar(mu, frac(e_g), yerr=ferr(e_g), fmt="-o", color=c["incorrect"], ms=4, capsize=0,
                  elinewidth=3, ecolor=mpl.colors.to_rgba(c["incorrect"], 0.3), label="incorrect GMM")
    ax_a.errorbar(mu, frac(e_s), yerr=ferr(e_s), fmt=":o", color=c["incorrect"], ms=2.5, capsize=0,
                  elinewidth=3, ecolor=mpl.colors.to_rgba(c["incorrect"], 0.3), label="incorrect SLOPE")
    ax_a.plot(mu, frac(n_mis), "-", color=c["missing"], label="missing")
    for m, eg, es, nt, p in zip(mu, e_g, e_s, n_tot, pvals):
        if p < P_SIGNIFICANT and es > 0 and eg / nt * 0.8 > cfg["ylim"][0]:
            ax_a.text(m * 1.08, eg / nt * 0.8, f"{100 * (1 - eg / es):.0f}%", color=c["incorrect"],
                      fontsize=8, ha="left", va="top")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_ylim(*cfg["ylim"])
    ax_a.set_xlabel(r"mean photon number $\mu$")
    ax_a.set_ylabel("fraction of frames")
    ax_a.grid(which="major", alpha=0.25)
    ax_a.grid(which="minor", alpha=0.08)
    ax_a.legend(frameon=False, loc="upper left", fontsize=8)
    ax_a.text(-0.12, 1.02, "a)", transform=ax_a.transAxes, fontsize=12, va="top")

    # ---- b) decoded images (GMM, stream 0)
    truth = canonical_truth(cfg)
    for k, (ax, db) in enumerate(zip(ax_b, cfg["image_dbs"])):
        ev = events[db]
        sym = np.full(len(truth), -1)
        s0 = ev["stream"] == 0
        sym[ev["frame"][s0]] = ev["gmm_pred"][s0]
        ok = sym == truth
        i = int(np.where(dbs == db)[0][0])
        ax.imshow(decoded_image(sym, cfg["bits"]), interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(MU_COLORS[k]), sp.set_linewidth(2.5)
        ax.set_title(rf"$\mu={mu_all[i]:.2f}$    {ok.mean() * raw_rate:.1f} Mbit/s", fontsize=8,
                     color=MU_COLORS[k], y=-0.16)
        ax_a.axvline(mu_all[i], color=MU_COLORS[k], ls="--", alpha=0.35, lw=1)
    ax_b[0].text(-0.2, 1.0, "b)", transform=ax_b[0].transAxes, fontsize=12, va="top")

    # ---- c, d) scatter of detections relative to the true slot
    for ax, db, label in [(ax_c, cfg["scatter_dbs"][0], "c)"), (ax_d, cfg["scatter_dbs"][1], "d)")]:
        ev = events[db]
        off = display_offset(cfg, db)
        x = ev["xy"][:, 0] - T * ev["true"] + off[0]
        y = ev["xy"][:, 1] - T * ev["true"] + off[1]
        g, s_ = ev["gmm_ok"], ev["slope_ok"]
        c1 = one_photon_center(cfg, db) + off
        kw = dict(s=2.5, rasterized=True, lw=0)
        for i in range(-7, 8):
            if i:
                ax.scatter(x[g] + i * T, y[g] + i * T, color=c["copies"], alpha=0.05, **kw)
        ax.scatter(x[g & s_], y[g & s_], color=c["both"], alpha=0.08, label="both methods correct", **kw)
        ax.scatter(x[g & ~s_], y[g & ~s_], color=c["fixed"], alpha=0.5,
                   label="GMM correct,\nSLOPE incorrect", **kw)
        ax.scatter(x[~g & s_], y[~g & s_], color=c["broken"], alpha=0.5,
                   label="GMM incorrect,\nSLOPE correct", **kw)
        (x0, x1), (y0, y1) = cfg["scatter_window"]
        ax.set_xlim(c1[0] + x0, c1[0] + x1)
        ax.set_ylim(c1[1] + y0, c1[1] + y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$t_A$ (ps)")
        ax.set_ylabel(r"$t_B$ (ps)")
        i = int(np.where(dbs == db)[0][0])
        k = list(cfg["image_dbs"]).index(db) if db in cfg["image_dbs"] else 0
        ax.text(0.04, 0.97, rf"$\mu = {mu_all[i]:.2f}$", transform=ax.transAxes, fontsize=12,
                color=MU_COLORS[k], va="top", weight="bold")
        ax.text(-0.18, 1.0, label, transform=ax.transAxes, fontsize=12, va="top")
        lg = ax.legend(frameon=False, loc="lower right", fontsize=7, markerscale=4)
        for h in lg.legend_handles:
            h.set_alpha(0.8)

    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    fig.savefig(cfg["out"] + "_light.pdf", dpi=300)
    fig.savefig(cfg["out"] + "_light.png", dpi=150)

    table = dict(
        mu_fit_A=A, mu_fit_used_db=dbs[mu_sel].tolist(), raw_rate_mbps=raw_rate,
        rows=[dict(db=int(d), mu=float(m), mu_vacuum=float(mv), n_total=int(nt), n_detected=int(nd),
                   gmm_errors=int(eg), slope_errors=int(es), mcnemar_p=float(p),
                   reduction=(float(1 - eg / es) if es else None), excluded=bool(fl))
              for d, m, mv, nt, nd, eg, es, p, fl in zip(dbs, full["mu"], mu_vac, full["n_tot"], full["n_det"],
                                                         full["e_g"], full["e_s"], full["pvals"], flag)],
    )
    with open(cfg["out"] + "_table.json", "w") as f:
        json.dump(table, f, indent=1)
    print(f"A={A:.4g}  raw rate {raw_rate:.2f} Mbit/s  -> {cfg['out']}_light.pdf")


if __name__ == "__main__":
    main()
