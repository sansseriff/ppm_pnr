"""
Experimental (branch ica-gmm): stability of the photon-number fractions in the number of gaussian components.

For the attribution dataset (group_db) of each demonstration, the photon-number fractions are computed for models
built with N components by
  - the published method (build_pnr_model.py: full-covariance GMM of the preamble, filtering, KL graph, Louvain)
  - the ICA-space method (ica_gmm.py: two ICA rounds, diagonal components with width cap, valley grouping);
    read from the sweep of ica_gmm_sweep.py (SIGMA=5)
each with the group weights and timing shift refit and joint slot + photon-number attribution, as in Fig. 6.

usage: uv run python -m analysis.model_stability [--recompute]
outputs: <dataset>/inter/ica_gmm/stability_published.json, figs/ica_gmm/model_stability.{pdf,png}
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import orjson
from sklearn.mixture import GaussianMixture

from analysis import style
from analysis.build_pnr_model import affinity, filter_components, stable_communities
from analysis.datasets import REPO, decode_path, get
from analysis.ica_gmm import groups_from
from analysis.make_fig_pnr import GROUP_COLORS
from analysis.pnr_refit import (flatten, fit_weights_and_shift, init_shift_grid, joint_attribution, load_events,
                                load_groups, load_stream_events, poisson_pmf)

NS_PUBLISHED = (10, 12, 16, 20, 25, 30, 35, 40, 45, 60)
SEEDS = (42, 7)
LABELS = ["1", "2", "3", "4", "5+"]


class Evaluator:
    """refit weights + shift on the group_db dataset and attribute every detection (as pnr_refit / Fig. 6)"""

    def __init__(self, cfg):
        self.cfg, self.T = cfg, cfg["slot_ps"]
        path = decode_path(cfg, cfg["group_db"])
        self.x, _, _ = load_events(path, self.T)
        self.ev = load_stream_events(path, self.T)
        span = np.ptp(self.x, axis=0)
        self.bg = -np.log(span[0] * span[1])

    def __call__(self, model_path):
        groups = load_groups(model_path)
        means, _, inner_w, _ = flatten(groups)
        center = np.full(2, (np.median(self.x, axis=0) - np.average(means, weights=inner_w, axis=0)).mean())
        v0 = init_shift_grid(self.x, groups, center, self.bg, half_width=600.0, step=5.0)
        pi, v, ll = fit_weights_and_shift(self.x, groups, v0, self.bg)
        J = len(groups)
        slot, num = joint_attribution(self.ev, groups, pi[:J] / pi[:J].sum(), v, self.T)
        return dict(frac=(np.bincount(num, minlength=J) / len(num)).tolist(), ll=ll,
                    acc=float(np.mean(slot == self.ev["true"])), k=int(len(means)))


def published_sweep(cfg, ev, out_dir):
    pts = np.asarray(orjson.loads(open(decode_path(cfg, cfg["model_db"]), "rb").read())["gm_data"]["counts"], float)
    rows = []
    for n in NS_PUBLISHED:
        for seed in SEEDS:
            gm = GaussianMixture(n_components=n, random_state=seed).fit(pts)
            m, c, w = filter_components(gm.means_, gm.covariances_, gm.weights_, 1000.0)
            try:
                parts, _, freq = stable_communities(affinity(m, c), 5, np.arange(0.5, 3.01, 0.05))
            except RuntimeError:
                print(f"N={n} seed={seed}: no 5-group partition", flush=True)
                continue
            groups = sorted(groups_from((m, c, w), parts), key=lambda g: g["center_of_mass"][1], reverse=True)
            path = out_dir / f"published_n{n}_seed{seed}.json"
            json.dump(groups, open(path, "w"))
            r = ev(path) | dict(n=n, seed=seed, louvain_freq=freq)
            rows.append(r)
            print(f"published N={n:2d} seed={seed:2d} comps={r['k']:2d} acc {r['acc']:.4f} "
                  f"fractions {np.round(100 * np.asarray(r['frac']), 1).tolist()}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="redo the published-method sweep")
    args = ap.parse_args()
    style.apply()
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for col, key in enumerate(["20", "10"]):
        cfg = get(key)
        d = cfg["inter"] / "ica_gmm"
        cache = d / "stability_published.json"
        if args.recompute or not cache.exists():
            json.dump(published_sweep(cfg, Evaluator(cfg), d), open(cache, "w"), indent=1)
        pub = json.load(open(cache))
        ica = json.load(open(d / "sweep_axis_s5.json"))["results"]
        mu = {r["db"]: r["mu"] for r in json.loads(open(cfg["table"]).read())["rows"]}[cfg["group_db"]]
        for row, (rows, name, nkey) in enumerate([(pub, "published method", "k"),
                                                   (ica, "ICA-space model", "k")]):
            a = axs[row, col]
            J = max(len(r["frac"]) for r in rows)
            pois = poisson_pmf(mu, J)[1:]
            pois = pois / pois.sum()
            labels = LABELS if J == 5 else LABELS[:J - 1] + [f"{J}+"]
            for j in range(J):
                for seed, mk in zip(SEEDS, ("o", "s")):
                    sel = [(r[nkey], 100 * r["frac"][j]) for r in rows if len(r["frac"]) == J and r["seed"] == seed]
                    if sel:
                        xs, ys = zip(*sorted(sel))
                        a.plot(xs, ys, mk + "-", color=GROUP_COLORS[j], ms=4, lw=1, alpha=0.8,
                               label=rf"$M_{{{labels[j]}}}$" if seed == SEEDS[0] else None)
                a.axhline(100 * pois[j], color=GROUP_COLORS[j], ls=":", lw=1)
            a.set_xscale("log")
            a.set_xticks([7, 10, 20, 30, 50]), a.set_xticklabels(["7", "10", "20", "30", "50"])
            a.set_ylim(0, 40)
            a.set_title(f"{cfg['tag']}, {cfg['group_db']} dB ($\\mu$ = {mu:.2f}): {name}", fontsize=10)
            a.set_ylabel("fraction of detections (%)")
            a.xaxis.set_minor_formatter(plt.NullFormatter())
            a.legend(ncol=J, fontsize=9, loc="upper right", frameon=False)
        axs[1, col].set_xlabel("gaussian components in the model")
    fig.suptitle("photon-number fractions vs model size (dotted: Poisson; circles / squares: two GMM seeds)",
                 fontsize=11)
    out = REPO / "ppm_20ghz" / "figs" / "ica_gmm"
    out.mkdir(parents=True, exist_ok=True)
    style.save(fig, out / "model_stability")


if __name__ == "__main__":
    main()
