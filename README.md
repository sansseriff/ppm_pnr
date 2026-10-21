# ppm_pnr

Analysis code for the manuscript

> A. Mueller, B. Korzh, E. Wollman, A. D. Beyer, I. Craiciu, M. D. Shaw, M. Spiropulu,
> **"Data Recovery and Pulse Position Modulation with a Photon Number Resolving SNSPD"**,
> Optics Express (submitted). DOI: *to be added*

A differential SNSPD is read out at two trigger levels (t_A, t_B). The pair (t_A, t_B) carries both the
arrival time and the photon number of each detection. This repository decodes 2048-ary pulse-position
modulation (PPM) at 20 GHz (50 ps slots) and 1024-ary PPM at 10.75 GHz (93 ps slots) with a slope correction
("SLOPE") and with a Gaussian-mixture model of the detector response ("GMM"), and it extracts photon-number
statistics from the same detections.

The repository holds **code only**. The data are on figshare, one item per subset, and
`figshare_download.py` puts them where the scripts expect them (see [Data](#data)).

## Quick start

```sh
# 1. environment (uv: https://docs.astral.sh/uv/). Python 3.12 is installed by uv if needed.
uv sync                                   # analysis + notebook dependencies
#   uv sync --no-default-groups           # scripts only, no Jupyter

# 2. data: the decoded subsets are enough for every figure (~140 MB of zips)
uv run python figshare_download.py --dataset ppm_20ghz_decoded ppm_10ghz_decoded waveform

# 3. figures
scripts/run_all.sh                 # Figs. 5-7, supplemental 10.75 GHz figures, SLOPE/GMM tables (~10 min)
scripts/run_all.sh --notebooks     # also Figs. 2, 3, 4, S1 (executes the notebooks)
```

Outputs go to `ppm_20ghz/figs/`, `ppm_10ghz/figs/`, `waveform/figs/` and the `inter/` folders.

## Data

| key (`--dataset`) | content | zip | unpacked | DOI (reserved) |
|---|---|---|---|---|
| `ppm_20ghz_decoded` | 20 GHz decode files (16-48 dB), photon-number model, PNR refit, 2023 alignment files | 27 MB | 148 MB | 10.6084/m9.figshare.34741485 |
| `ppm_10ghz_decoded` | 10.75 GHz decode files (18-44 dB, 42 dB = corrected re-decode), photon-number model, PNR refit | 114 MB | 512 MB | 10.6084/m9.figshare.34741488 |
| `waveform` | trigger-level scan of the detector pulse (Fig. 2), t_B - t_A delay | 0.1 MB | 3 MB | 10.6084/m9.figshare.34741491 |
| `ppm_20ghz_raw` | raw Swabian Time Tagger files (16-50 dB) + AWG ground truth | 2.0 GB | 2.9 GB | 10.6084/m9.figshare.34741494 |
| `ppm_10ghz_raw` | raw Swabian Time Tagger files (18-44 dB) + AWG ground truth | 1.3 GB | 2.1 GB | 10.6084/m9.figshare.34741497 |

Figshare project: *link to be added on publication* (the items are private drafts until the paper is accepted).

```sh
uv run python figshare_download.py --list                         # sizes, DOIs, what is already here
uv run python figshare_download.py --dataset ppm_10ghz_decoded    # any subset
uv run python figshare_download.py --all                          # everything (3.4 GB of zips)
uv run python figshare_download.py                                # interactive menu
```

Each zip contains paths relative to the repository root (for example
`ppm_20ghz/inter/decode_20GHz24.0.json`) and is unpacked in place. The download is checked against the md5
that figshare computed on upload, and every unpacked file against the dataset's `MANIFEST_<key>.txt`.
Datasets that are already present are skipped (`--force` downloads again, `--verify` re-checks md5s).
Until the items are published, set `FIGSHARE_TOKEN` to a personal figshare token with access to them.
The raw data are only needed to re-run the decoder.

All dataset metadata (article ids, DOIs, sizes, file lists) are in `figshare_metadata.yaml`.

## Layout

```
analysis/                  dataset-agnostic scripts, all take --dataset 10|20
  datasets.py              per-dataset settings (attenuations, slot width, GMM index, panels) -- edit here
  load_schema.py           pydantic-v2 loaders for the decode files
  fair_comparison.py       SLOPE vs GMM with cross-validated offsets + McNemar test (supplemental tables)
  make_fig_decoding.py     decoding-performance figure (Fig. 5; 10.75 GHz: supplement)
  build_pnr_model.py       photon-number model (groups of GMM components) from a preamble fit
  pnr_refit.py             prior-free photon-number statistics (EM for group weights + common-mode shift)
  make_fig_pnr.py          photon-number attribution and statistics figures (Figs. 6, 7; 10.75 GHz: supplement)
ppm_20ghz/
  src_decode/              decoder: raw .ttbin -> inter/decode_20GHz{dB}.0.json (Linux container, see below)
  notebooks/               notebooks that are still the only source of a figure (Figs. 3, 4, S1)
  deprecated/              superseded 2023 notebooks and their helper modules, outputs stripped, not maintained
  data/  inter/  figs/     filled by figshare_download.py / the scripts (git-ignored)
ppm_10ghz/
  src_decode/  (+ docker/) decoder and the Linux container used for both datasets
  deprecated/              superseded 2023 notebooks
  data/  inter/  figs/
waveform/
  notebooks/main.ipynb     Fig. 2
  data/  figs/
scripts/run_all.sh         regenerate everything from the decoded data
figshare_download.py       data downloader (users)
figshare_metadata.yaml     figshare ids, DOIs, sizes, file lists of every dataset
tools/                     maintainer tools: package_datasets.py (build the zips), figshare_upload.py (upload)
```

Run the scripts from the repository root as modules (`uv run python -m analysis.<name> --dataset 20`), or
import them (`from analysis.fair_comparison import analyze`); nothing runs at import time.

## Figures

| Figure | Made by | Inputs |
|---|---|---|
| Fig. 1 (setup) | not code (drawn in Blender/Illustrator) | |
| Fig. 2 waveform | `waveform/notebooks/main.ipynb` | `waveform` |
| Fig. 3 slope correction | `ppm_20ghz/notebooks/decode_analysis.ipynb` | `ppm_20ghz_decoded` (22, 26, 30 dB) |
| Fig. 4 GMM introduction | `ppm_20ghz/notebooks/gmm_intro_analysis_t.ipynb` | `ppm_20ghz_decoded` (26, 36 dB, `offset_correction_*.json`) |
| Fig. 5 PPM decoding | `python -m analysis.make_fig_decoding --dataset 20` | `ppm_20ghz_decoded` |
| Fig. 6 photon-number attribution | `python -m analysis.make_fig_pnr --dataset 20` | `static_sim.json`, `pnr_refit.json`, Fig. 5 table |
| Fig. 7 photon statistics | `python -m analysis.make_fig_pnr --dataset 20` | as Fig. 6 |
| Fig. S1 ICA | `ppm_20ghz/notebooks/gmm_intro_analysis_t.ipynb` | 24 dB, `notebooks/vec.json` |
| Tables S (SLOPE vs GMM, 20 and 10.75 GHz) | `python -m analysis.fair_comparison --dataset 20` / `10` -> `inter/fair_comparison.json`; also `figs/ppm_decoding_<tag>_table.json` | decoded data |
| 10.75 GHz supplemental figures | `make_fig_decoding` / `make_fig_pnr --dataset 10` | `ppm_10ghz_decoded` |

`pnr_refit.json` is regenerated by `python -m analysis.pnr_refit --dataset 20|10` (run_all.sh does this; the
downloaded copy is identical). The 20 GHz scripts write `pnr_*_20GHz_light.pdf`; the manuscript includes
them as `pnr_resolved_arrival_time_groupings_light.pdf` and `pnr_stats_light.pdf`.

The notebooks use LaTeX for a few bold labels if `latex` is installed and fall back to matplotlib mathtext
otherwise (the figure content is the same).

## Pipeline

```
 raw .ttbin + ground_truth.zip                         (figshare *_raw)
        |  src_decode/JitRead_*.py  (Linux container: Swabian TimeTagger, sklearn 1.2.2)
        |    software PLL on the AWG clock, pairing of t_A/t_B, preamble slope-correction lookup,
        |    23 (20 GHz) / 10 (10 GHz) GaussianMixture fits of the preamble, frame/slot assignment
        v
 inter/decode_<f>GHz{dB}.0.json                        (figshare *_decoded)
        |------------------------------------------------------------.
        |  analysis/fair_comparison.py                                |  analysis/build_pnr_model.py
        |    SLOPE and GMM decisions, CV offsets, McNemar             |    filter components, KL graph,
        v                                                             |    Louvain -> photon groups C1..C5+
 inter/fair_comparison.json, inter/fair_events_*.npz                  v
        |  analysis/make_fig_decoding.py                       inter/static_sim.json
        v                                                             |  analysis/pnr_refit.py
 figs/ppm_decoding_*  (Fig. 5) + *_table.json (mu calibration) ------>|    EM: group weights + shift
                                                                      v
                                                               inter/pnr_refit.json
                                                                      |  analysis/make_fig_pnr.py
                                                                      v
                                                    figs/pnr_resolved_arrival_time_groupings_*, pnr_stats_*
                                                                      (Figs. 6, 7)
```

## Decoding the raw data (optional)

The decoder reads the `.ttbin` files with the Swabian Time Tagger API, which has no macOS build, so it runs
in a Linux container. One image serves both datasets: `ppm_10ghz/src_decode/docker/`
(Python 3.11, `Swabian-TimeTagger` 2.22.6 from PyPI, numpy 1.26.4, scikit-learn 1.2.2).

```sh
uv run python figshare_download.py --dataset ppm_20ghz_raw ppm_10ghz_raw
colima start --cpu 6 --memory 16 --mount "$PWD:w"        # macOS only (brew install colima docker)
docker build -t ppm-pnr-decode ppm_10ghz/src_decode/docker
docker run --rm -v "$PWD":/work -w /work/ppm_20ghz/src_decode ppm-pnr-decode python redecode.py --all
docker run --rm -v "$PWD":/work -w /work/ppm_10ghz/src_decode ppm-pnr-decode python redecode.py --all
```

Outputs go to `<dataset>/inter/redecode/`, so the downloaded files are not overwritten. With this image the
re-decode reproduces the published decode files event for event (all five repetitions, every field of every
frame); the stored GMM parameters agree to ~1e-8 relative (floating-point differences between platforms).
scikit-learn is pinned to 1.2.2 on purpose: later versions converge to slightly different GMM optima.
Details: `ppm_10ghz/src_decode/docker/README.md`.

## Environment

One uv environment (`pyproject.toml`, `uv.lock`, Python 3.12) for all analysis:
numpy >= 2.1 (2.0.x gives spurious matmul warnings with macOS Accelerate), scipy, scikit-learn, numba,
matplotlib, orjson, networkx, pillow, pydantic 2, pyyaml, requests; Jupyter, `snsphd` (the author's plotting
helpers), bokeh and plotly in the default `notebooks` group; `ruamel.yaml` and `nbstripout` in the `tools`
group. The 2023 code used pydantic 1.9; `analysis/load_schema.py` is the pydantic-v2 port
(`@validator` -> annotated numpy field type, `class Config` -> `model_config`).

The analysis results do not depend on the package versions beyond floating-point rounding: with the locked
environment, every count in the SLOPE/GMM tables and the photon-number fits reproduces the published values
exactly, and the floating-point values agree to better than 1e-10 relative.

## What is not here

- the 2023 exploration notebooks are in `*/deprecated/` (outputs stripped, code unchanged, not maintained);
  they need files that are not on figshare (`inter/output.json`, `inter/output_pnr.json`, ~285 MB, made by
  `deprecated/gmm_effectiveness_analysis*.ipynb` with `recompute=True`)
- the Bokeh/Plotly web visualisations (`www/`, published at https://sansseriff.github.io/ppm_20ghz/),
  old intermediate folders (`inter/old*`), the `testing/` raw files of the 20 GHz run, and Illustrator files

## License and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0. Please cite the article (see `CITATION.cff`).
