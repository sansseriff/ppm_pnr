#!/usr/bin/env bash
# Regenerate the figures and tables of the paper from the decoded data.
#
#   scripts/run_all.sh                    # every figure of the paper and supplement + SLOPE/GMM tables
#   scripts/run_all.sh --notebooks        # also execute the walk-through notebooks in notebooks/
#   scripts/run_all.sh --rebuild-models   # also rebuild the unconstrained models (-> inter/static_sim_rebuilt.json)
#   scripts/run_all.sh --stability        # also the model-size sweeps of the stability figure (about 1 h)
#   scripts/run_all.sh --dataset 20       # only one dataset (10 or 20)
#
# Needs the decoded data:  uv run python figshare_download.py --dataset ppm_20ghz_decoded ppm_10ghz_decoded waveform
# Outputs (per dataset folder ppm_20ghz/, ppm_10ghz/):
#   inter/fair_comparison.json            SLOPE vs the decoder's own (unconstrained) GMM; also writes the mu table
#   figs/ppm_decoding_<tag>_table.json    calibrated mu (used by everything below)
#   inter/ica_gmm/model_n20_cap5.json (+ _geometry.json)   photon-number-structured model (ICA feature space)
#   inter/ica_gmm/pnr_refit_n20_cap5.json                  group weights + offset per attenuation (Figs. 7, 8)
#   inter/fair_comparison_local.json, figs/ica_gmm/ppm_decoding_<tag>_light.*   SLOPE vs structured GMM (Fig. 6)
#   figs/table_fair_<tag>.tex                              Tables S1, S2
#   figs/pnr_resolved_arrival_time_groupings_<tag>_light.*, figs/pnr_stats_<tag>_light.*   Figs. 7, 8
# Takes about 30 min per dataset on an M-series Mac.
set -euo pipefail
cd "$(dirname "$0")/.."

NOTEBOOKS=0
REBUILD=0
STABILITY=0
DATASETS="20 10"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --notebooks) NOTEBOOKS=1 ;;
    --rebuild-models) REBUILD=1 ;;
    --stability) STABILITY=1 ;;
    --dataset) DATASETS="$2"; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
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
  mkdir -p "$dir/inter/ica_gmm" "$dir/figs/ica_gmm"
  run uv run python -m analysis.fair_comparison --dataset "$ds" --recompute
  run uv run python -m analysis.make_fig_decoding --dataset "$ds"
  run uv run python -m analysis.ica_gmm --dataset "$ds" --rounds 2 --n 20 --grouping axis --sigma-max 5 \
      --out "$dir/inter/ica_gmm/model_n20_cap5.json"
  run uv run python -m analysis.pnr_refit --dataset "$ds"
  run uv run python -m analysis.fair_comparison --dataset "$ds" --model local --recompute
  run uv run python -m analysis.make_fig_decoding --dataset "$ds" --model local
  run uv run python -m analysis.make_fig_pnr --dataset "$ds"
  if [[ $STABILITY == 1 ]]; then
    run env DS="$ds" GROUPING=axis SIGMA=5 NS="4 6 8 10 12 16 20 25 30 40" uv run python -m analysis.ica_gmm_sweep
  fi
  if [[ $REBUILD == 1 ]]; then
    run uv run python -m analysis.build_pnr_model --dataset "$ds"
  fi
done

# figures that need the 20 GHz mu table and fitted offsets written above, or the waveform scan
if [[ " $DATASETS " == *" 20 "* ]]; then
  run uv run python -m analysis.make_fig_slope                                           # Fig. 3
  run uv run python -m analysis.make_fig_ica_space                                       # Fig. 4
  run uv run python -m analysis.make_fig_structured_gmm                                  # Fig. 5
  run uv run python -m analysis.make_fig_ica                                             # Fig. S1
fi
run uv run python -m analysis.make_tables                                                 # Tables S1, S2
if [[ $STABILITY == 1 ]]; then
  run uv run python -m analysis.model_stability                                          # Fig. S(stability)
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
