#!/usr/bin/env bash
# Regenerate the figures and tables of the paper from the decoded data.
#
#   scripts/run_all.sh                    # every figure of the paper and supplement + SLOPE/GMM tables
#   scripts/run_all.sh --notebooks        # also execute the walk-through notebooks in notebooks/
#   scripts/run_all.sh --rebuild-models   # also rebuild the photon-number models (-> inter/static_sim_rebuilt.json)
#   scripts/run_all.sh --dataset 20       # only one dataset (10 or 20)
#
# Needs the decoded data:  uv run python figshare_download.py --dataset ppm_20ghz_decoded ppm_10ghz_decoded waveform
# Outputs (per dataset folder ppm_20ghz/, ppm_10ghz/):
#   inter/fair_comparison.json, inter/fair_events_*.npz   SLOPE vs GMM comparison (supplemental tables)
#   figs/ppm_decoding_<tag>_light.{pdf,png}, figs/ppm_decoding_<tag>_table.json         Fig. 5 / Fig. S(10 GHz)
#   inter/pnr_refit.json (overwrites the downloaded copy with an identical recomputation)
#   figs/pnr_resolved_arrival_time_groupings_<tag>_light.{pdf,png}, figs/pnr_stats_<tag>_light.{pdf,png},
#   figs/pnr_figures_<tag>.json                                                          Figs. 6, 7 / Figs. S(10 GHz)
# Takes about 5 min per dataset on an M-series Mac.
set -euo pipefail
cd "$(dirname "$0")/.."

NOTEBOOKS=0
REBUILD=0
DATASETS="20 10"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --notebooks) NOTEBOOKS=1 ;;
    --rebuild-models) REBUILD=1 ;;
    --dataset) DATASETS="$2"; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

run() { echo "+ $*"; "$@"; }

for ds in $DATASETS; do
  case "$ds" in
    20) dir=ppm_20ghz; first=decode_20GHz16.0.json ;;
    10) dir=ppm_10ghz; first=decode_10GHz18.0.json ;;
    *) echo "unknown dataset $ds" >&2; exit 2 ;;
  esac
  if [[ ! -f "$dir/inter/$first" ]]; then
    echo "!! $dir/inter has no decoded data; run: uv run python figshare_download.py --dataset ${dir}_decoded" >&2
    exit 1
  fi
  echo "=== dataset $ds ($dir)"
  run uv run python -m analysis.fair_comparison --dataset "$ds" --recompute
  run uv run python -m analysis.make_fig_decoding --dataset "$ds"
  run uv run python -m analysis.pnr_refit --dataset "$ds"
  run uv run python -m analysis.make_fig_pnr --dataset "$ds"
  if [[ $REBUILD == 1 ]]; then
    run uv run python -m analysis.build_pnr_model --dataset "$ds"
  fi
done

# figures that need the 20 GHz mu table and fitted offsets written above, or the waveform scan
if [[ " $DATASETS " == *" 20 "* ]]; then
  run uv run python -m analysis.make_fig_slope                                           # Fig. 3
  run uv run python -m analysis.make_fig_gmm_intro                                       # Fig. 4
  run uv run python -m analysis.make_fig_ica                                             # Fig. S1
fi
if [[ -f waveform/data/decent.npy ]]; then
  run uv run python -m analysis.make_fig_waveform                                        # Fig. 2
fi

if [[ $NOTEBOOKS == 1 ]]; then
  # walk-through notebooks (they only call analysis/ and plot); executed copies go to a temp dir
  out=$(mktemp -d)
  for nb in notebooks/decoding_comparison.ipynb notebooks/photon_number_model.ipynb notebooks/ica_contrast.ipynb; do
    run uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --output-dir "$out" "$nb"
  done
  echo "executed notebooks in $out"
fi
echo "done"
