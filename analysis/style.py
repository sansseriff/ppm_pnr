"""
Matplotlib style shared by the figure scripts: snsphd.viz.phd_style with the arguments of the original
notebooks. save() writes the light version exactly as snsphd.viz.save_light_dark_all did (its other
options only affect the dark version, which the paper does not use).

Axis labels, ticks and legends use DejaVu Sans. A few callouts (photon-number kets, the mu and component
labels inside panels, slot labels) are typeset by LaTeX with \\bm, as the notebooks did with usetex=True
on those texts only. LaTeX is used when latex and dvipng are found on PATH or in ~/Library/TinyTeX;
otherwise those callouts fall back to bold mathtext, so every script runs without TeX.

    from analysis import style
    style.apply()
    ax.text(x, y, style.ket("1"), usetex=style.USETEX)
    style.save(fig, figs / "slope_cancellation")  # -> slope_cancellation_light.{pdf,png}
"""

import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
from snsphd import viz

TINYTEX_BIN = Path.home() / "Library" / "TinyTeX" / "bin" / "universal-darwin"


def have_latex() -> bool:
    if TINYTEX_BIN.is_dir() and str(TINYTEX_BIN) not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + str(TINYTEX_BIN)
    return all(shutil.which(b) for b in ("latex", "dvipng"))


USETEX = have_latex()


def apply():
    viz.phd_style(jupyterStyle=True, grid=True, axese_width=0.5, text=6, svg_mode=True)
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{bm}"


def bold(math: str) -> str:
    """bold math callout; pass usetex=USETEX to the text call"""
    return rf"$\bm{{{math}}}$" if USETEX else rf"$\mathbf{{{math}}}$"


def ket(n) -> str:
    return bold(rf"|{n}\rangle")


def save(fig, base: Path, png_dpi: int = 110):
    """<base>_light.pdf (and a .png preview), as snsphd.viz.save_light_dark_all saves the light version"""
    for ax in fig.axes:
        ax.set_facecolor(mcolors.to_rgba("#FDFEFF", 0.60))
    fig.patch.set_facecolor("white")
    fig.savefig(f"{base}_light.pdf")
    fig.savefig(f"{base}_light.png", dpi=png_dpi)
    print(f"-> {base}_light.pdf")
