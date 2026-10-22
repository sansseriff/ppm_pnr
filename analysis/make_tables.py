"""
Tables S1, S2: SLOPE vs GMM decoding per attenuation, as LaTeX tabular rows (body only; the table environment and
caption are in the supplement).

GMM = the photon-number-structured model refitted to each attenuation's preamble (fair_comparison.py --model local).
mu is the calibrated mean photon number of make_fig_decoding.py.

usage: uv run python -m analysis.make_tables   -> <dataset>/figs/table_fair_<tag>.tex
"""

import json

from analysis.datasets import get


def p_tex(p):
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def mu_tex(m):
    return f"{m:.3g}"


def main():
    for key in ("20", "10"):
        cfg = get(key)
        mu = {r["db"]: r["mu"] for r in json.load(open(cfg["table"]))["rows"]}
        res = json.load(open(cfg["inter"] / "fair_comparison_local.json"))
        lines = []
        for db in sorted(res, key=float):
            r = res[db]
            red = 100 * (1 - r["gmm_errors"] / r["slope_errors"]) if r["slope_errors"] else float("nan")
            lines.append(f"{float(db):.0f} & {mu_tex(mu[int(float(db))])} & {r['n_detected']} & {r['slope_errors']} & "
                         f"{r['gmm_errors']} & {red:.1f} & {p_tex(r['mcnemar_p'])} \\\\")
        out = cfg["figs"] / f"table_fair_{cfg['tag']}.tex"
        out.write_text("\n".join(lines) + "\n")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
