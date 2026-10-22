"""
Fair comparison of SLOPE vs GMM PPM decoding (supplemental tables; input to Fig. 5 / Fig. S-decoding).

Differences from worker.caller():
  - both methods get their own timing-offset sweep (worker.py only swept the GMM offset;
    the slope sweep always used offset=0, so its accuracy curve was flat)
  - offsets are chosen on held-out streams (cross-validation) instead of on the events being scored
  - all 5 decoded streams per attenuation are used, not just streams 0-1
  - GMM likelihoods are evaluated in log space, so tags far from every component don't underflow
    to p=0 for all candidates (np.argmax of all-zeros silently picks the -1 bin)
  - significance of GMM vs SLOPE is assessed with an exact McNemar test on the paired decisions

GMM models (--model):
  decoder     the GMM the decoder fitted to this attenuation's preamble (gm_list[cfg["gm_idx"]])
  structured  the photon-number-structured model (ICA feature space, analysis/ica_gmm.py; cfg["structured_model"])
              with its group weights, a uniform background and the common-mode timing shift fitted by EM to this
              attenuation's preamble (slot 0 by construction). The transmitted data, and its true slots, are not
              used for the calibration.
  local       the structured model's ICA bases, |1> cut and valley boundaries, with the components refitted to
              this attenuation's preamble (ica_gmm.local_refit), then weights / background / shift as above.
              The photon-number groups keep their meaning; the component shapes follow the photon-number
              distribution of each attenuation, as the decoder's own GMM does.

Per-event results are cached in <dataset>/inter/fair_events_<model prefix><prefix><dB>.npz (used by
make_fig_decoding.py).

usage: uv run python -m analysis.fair_comparison --dataset 20   (or 10)   [--model structured] [--recompute]
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import orjson
from scipy.stats import binomtest
from scipy.special import logsumexp

from analysis.datasets import decode_path, get
from analysis.load_schema import Decode, Result
from analysis.ica_gmm import geometry_path, local_refit
from analysis.pnr_refit import fit_weights_and_shift, flatten, init_shift_grid, load_groups

OFFSETS = np.arange(-0.25, 0.25001, 0.0025)  # in units of bins
CANDIDATES = np.array([-1, 0, 1])  # GMM bins compared relative to the slope estimate


def load_decode(path: str) -> Decode:
    with open(path, "rb") as f:
        return Decode(**orjson.loads(f.read()))


def event_arrays(decode: Decode):
    """Flatten non-missing events of every stream into arrays. Deadtime / extra-tag events are kept
    as errors for both methods (they are the same for both and contribute equally)."""
    rows = []
    for stream_idx, stream in enumerate(decode.results):
        for frame_idx, e in enumerate(stream):
            if e.result == Result.MISSING:
                continue
            valid = e.result in (Result.CORRECT, Result.INCORRECT)
            rows.append((stream_idx, e.tag, e.tag_x, e.tag_y, e.true, e.measured, valid, frame_idx))
    a = np.array(rows, dtype=float)
    n_missing = sum(e.result == Result.MISSING for s in decode.results for e in s)
    return dict(
        stream=a[:, 0].astype(int),
        tag=a[:, 1],
        xy=a[:, 2:4],
        true=a[:, 4].astype(int),
        measured=a[:, 5].astype(int),
        valid=a[:, 6].astype(bool),
        frame=a[:, 7].astype(int),
        n_missing=n_missing,
        n_total=n_missing + len(a),
    )


def gmm_logpdf(xy: np.ndarray, gm) -> np.ndarray:
    """log p(xy) for a 2D gaussian mixture. xy: (N, 2) -> (N,)"""
    means = np.asarray(gm.means)
    covs = np.asarray(gm.covariances)
    weights = np.asarray(gm.weights)
    inv = np.linalg.inv(covs)
    logdet = np.linalg.slogdet(covs)[1]
    d = xy[:, None, :] - means[None, :, :]  # (N, K, 2)
    maha = np.einsum("nki,kij,nkj->nk", d, inv, d)
    comp = -0.5 * maha - 0.5 * logdet - np.log(2 * np.pi) + np.log(weights)
    return logsumexp(comp, axis=1)


def calibrate_structured(pre, model_path, local=False):
    """the structured model calibrated on preamble detections pre: (groups, weights, shift, background fraction).
    The group weights, a uniform background and the common-mode timing shift are fitted by EM. local: the components
    are first refitted to pre in the model's ICA bases (ica_gmm.local_refit). groups are in the model's time frame
    (load_groups format); the shift maps them onto pre."""
    groups = load_groups(model_path)
    means, _, inner_w, _ = flatten(groups)
    span = np.ptp(pre, axis=0)
    bg = -np.log(span[0] * span[1])
    center = np.full(2, (np.median(pre, axis=0) - np.average(means, weights=inner_w, axis=0)).mean())
    v0 = init_shift_grid(pre, groups, center, bg, half_width=600.0, step=5.0)
    pi, v, _ = fit_weights_and_shift(pre, groups, v0, bg)
    if local:
        geometry = json.load(open(geometry_path(model_path)))
        path = os.path.join(os.path.dirname(model_path), f"_local_tmp_{os.getpid()}.json")
        json.dump(local_refit(pre - v, geometry, json.load(open(model_path))), open(path, "w"))
        groups = load_groups(path)
        os.remove(path)
        pi, dv, _ = fit_weights_and_shift(pre - v, groups, np.zeros(2), bg)
        v = v + dv
    J = len(groups)
    return groups, pi[:J] / pi[:J].sum(), v, float(pi[J])


def structured_gm(decode: Decode, model_path, local=False):
    """the structured model calibrated on this attenuation's preamble: (gm for gmm_logpdf, calibration info)"""
    groups, w, v, bg_frac = calibrate_structured(np.asarray(decode.gm_data.counts, float), model_path, local)
    means, covs, inner_w, label = flatten(groups)
    gm = SimpleNamespace(means=means + v, covariances=covs, weights=inner_w * w[label])
    return gm, dict(preamble_weights=w.tolist(), preamble_shift=v.tolist(), preamble_background=bg_frac)


def slope_predict(ev, offset_bins: float, T: float) -> np.ndarray:
    return np.round((ev["tag"] + offset_bins * T) / T).astype(int)


def gmm_loglik_table(ev, gm, T: float, offset_bins: float) -> np.ndarray:
    """(N, len(CANDIDATES)) log-likelihoods of each event coming from bin measured+k."""
    out = np.empty((len(ev["tag"]), len(CANDIDATES)))
    for j, k in enumerate(CANDIDATES):
        shift = (ev["measured"] + k + offset_bins) * T
        out[:, j] = gmm_logpdf(ev["xy"] - shift[:, None], gm)
    return out


def gmm_predict(ev, gm, T: float, offset_bins: float) -> np.ndarray:
    ll = gmm_loglik_table(ev, gm, T, offset_bins)
    return ev["measured"] + CANDIDATES[np.argmax(ll, axis=1)]


def correct(ev, pred):
    return (pred == ev["true"]) & ev["valid"]


def best_offset(ev, mask, predict_fn):
    scores = [np.sum(correct(ev, predict_fn(o))[mask]) for o in OFFSETS]
    return float(OFFSETS[int(np.argmax(scores))])


def mcnemar(gmm_ok, slope_ok):
    b = int(np.sum(gmm_ok & ~slope_ok))  # fixed by GMM
    c = int(np.sum(~gmm_ok & slope_ok))  # broken by GMM
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p


def bootstrap_reduction(gmm_err, slope_err, n=2000, seed=0):
    """95% CI on relative error-rate reduction 1 - E_gmm/E_slope, resampling events (paired)."""
    rng = np.random.default_rng(seed)
    N = len(gmm_err)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        es = slope_err[idx].sum()
        if es > 0:
            vals.append(1 - gmm_err[idx].sum() / es)
    return np.percentile(vals, [2.5, 97.5]).tolist()


def analyze(decode: Decode, gm_idx: int, laser_time_ps: float | None = None, keep_events: bool = False,
            structured_model=None, local=False):
    """keep_events: also return per-event arrays (under '_events') for plotting; not JSON-serialisable.
    structured_model: path of a structured model; used instead of the decoder's gm_list[gm_idx]"""
    T = laser_time_ps if laser_time_ps is not None else decode.set_data["laser_time"] * 1e12
    calib = {}
    if structured_model is None:
        gm = decode.gm_data.gm_list[gm_idx]
    else:
        gm, calib = structured_gm(decode, structured_model, local=local)
    ev = event_arrays(decode)
    streams = np.unique(ev["stream"])

    # precompute GMM predictions over the offset grid once (the expensive part)
    gmm_preds = {o: gmm_predict(ev, gm, T, o) for o in OFFSETS}
    slope_pred = lambda o: slope_predict(ev, o, T)
    gmm_pred = lambda o: gmm_preds[o]

    # leave-one-stream-out: tune offsets on the other streams, score on this one
    gmm_ok = np.zeros(len(ev["tag"]), bool)
    slope_ok = np.zeros(len(ev["tag"]), bool)
    gmm_cv_pred = np.zeros(len(ev["tag"]), int)
    slope_cv_pred = np.zeros(len(ev["tag"]), int)
    cv_offsets = []
    for s in streams:
        test = ev["stream"] == s
        o_g = best_offset(ev, ~test, gmm_pred)
        o_s = best_offset(ev, ~test, slope_pred)
        cv_offsets.append((int(s), o_g, o_s))
        gmm_cv_pred[test] = gmm_pred(o_g)[test]
        slope_cv_pred[test] = slope_pred(o_s)[test]
    gmm_ok = correct(ev, gmm_cv_pred)
    slope_ok = correct(ev, slope_cv_pred)

    # adjacent-slot (timing) errors are the ones a decoding method can influence; gross errors
    # (deadtime/extra tags, dark counts, sequence misalignment) are lumped separately
    adj_g = ev["valid"] & (np.abs(gmm_cv_pred - ev["true"]) == 1)
    adj_s = ev["valid"] & (np.abs(slope_cv_pred - ev["true"]) == 1)

    # reference numbers: legacy scheme (GMM offset tuned in-sample, slope offset fixed at 0)
    all_mask = np.ones(len(ev["tag"]), bool)
    o_g_in = best_offset(ev, all_mask, gmm_pred)
    o_s_in = best_offset(ev, all_mask, slope_pred)
    legacy_gmm_err = int(np.sum(~correct(ev, gmm_pred(o_g_in))))
    legacy_slope_err = int(np.sum(~correct(ev, slope_pred(0.0))))

    n_det = len(ev["tag"])
    gmm_err = ~gmm_ok
    slope_err = ~slope_ok
    b, c, p = mcnemar(gmm_ok, slope_ok)
    E_g, E_s = int(gmm_err.sum()), int(slope_err.sum())
    out = dict(
        laser_time_ps=T,
        n_total=ev["n_total"],
        n_missing=ev["n_missing"],
        n_detected=n_det,
        n_invalid=int(np.sum(~ev["valid"])),
        mu_raw=float(-np.log(ev["n_missing"] / ev["n_total"])),
        cv_offsets=cv_offsets,
        insample_offset_gmm=o_g_in,
        insample_offset_slope=o_s_in,
        gmm_errors=E_g,
        slope_errors=E_s,
        gmm_error_rate=E_g / n_det,
        slope_error_rate=E_s / n_det,
        reduction=(1 - E_g / E_s) if E_s else None,
        reduction_ci95=bootstrap_reduction(gmm_err, slope_err) if E_s else None,
        gmm_adjacent_errors=int(adj_g.sum()),
        slope_adjacent_errors=int(adj_s.sum()),
        adjacent_reduction=(1 - adj_g.sum() / adj_s.sum()) if adj_s.sum() else None,
        adjacent_reduction_ci95=bootstrap_reduction(adj_g, adj_s) if adj_s.sum() else None,
        fixed_by_gmm=b,
        broken_by_gmm=c,
        mcnemar_p=p,
        legacy_gmm_errors=legacy_gmm_err,
        legacy_slope_errors=legacy_slope_err,
        **calib,
    )
    if keep_events:
        out["_events"] = dict(
            ev, gmm_pred=gmm_cv_pred, slope_pred=slope_cv_pred, gmm_ok=gmm_ok, slope_ok=slope_ok
        )
    return out


MODEL_PREFIX = dict(decoder="", structured="structured_", local="local_")


def cache_path(cfg, db, model="decoder"):
    return os.path.join(cfg["inter"], f"fair_events_{MODEL_PREFIX[model]}{cfg['prefix']}{float(db):.1f}.npz")


def cached_analysis(cfg, db, recompute=False, model="decoder"):
    """analyze() one attenuation, keeping the per-event arrays; cached as an npz next to the decode file."""
    cache = cache_path(cfg, db, model)
    if os.path.exists(cache) and not recompute:
        z = np.load(cache, allow_pickle=True)
        return z["summary"].item(), {k: z[k] for k in z.files if k != "summary"}
    r = analyze(load_decode(decode_path(cfg, db)), cfg["gm_idx"], cfg["laser_time_ps"], keep_events=True,
                structured_model=cfg["structured_model"] if model in ("structured", "local") else None,
                local=model == "local")
    ev = r.pop("_events")
    arrays = {k: v for k, v in ev.items() if isinstance(v, np.ndarray)}
    r["n_streams"] = int(ev["stream"].max() + 1)
    np.savez_compressed(cache, summary=np.array(r, dtype=object), **arrays)
    return r, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["10", "20"], required=True)
    ap.add_argument("--db", nargs="*", type=float, default=None, help="default: all attenuations")
    ap.add_argument("--recompute", action="store_true", help="ignore cached fair_events_*.npz")
    ap.add_argument("--model", choices=list(MODEL_PREFIX), default="decoder")
    ap.add_argument("--out", default=None,
                    help="default: <dataset>/inter/fair_comparison.json (structured: fair_comparison_structured.json)")
    args = ap.parse_args()
    cfg = get(args.dataset)

    results = {}
    for db in args.db or cfg["dbs"]:
        r, _ = cached_analysis(cfg, db, recompute=args.recompute, model=args.model)
        r = {k: v for k, v in r.items() if k != "n_streams"}
        results[f"{db:.1f}"] = r
        red = f"{100*r['reduction']:.1f}%" if r["reduction"] is not None else "n/a"
        print(
            f"{db:4.0f} dB  mu={r['mu_raw']:.3f}  N={r['n_detected']:6d}  "
            f"slope={r['slope_errors']:4d}  gmm={r['gmm_errors']:4d}  red={red:>7}  "
            f"adj s/g={r['slope_adjacent_errors']}/{r['gmm_adjacent_errors']}  "
            f"b/c={r['fixed_by_gmm']}/{r['broken_by_gmm']}  p={r['mcnemar_p']:.2g}  "
            f"legacy s/g={r['legacy_slope_errors']}/{r['legacy_gmm_errors']}  "
            f"off_in g/s={r['insample_offset_gmm']:+.4f}/{r['insample_offset_slope']:+.4f}",
            flush=True,
        )

    default_out = cfg["fair"] if args.model == "decoder" else cfg["inter"] / f"fair_comparison_{args.model}.json"
    with open(args.out or default_out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
