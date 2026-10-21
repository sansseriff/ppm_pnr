# Decoding environment (Linux container, both datasets)

The decoders (`ppm_20ghz/src_decode/JitRead_PPMSets_20GHz.py`, `ppm_10ghz/src_decode/JitRead_PPMSets.py`)
read the raw `.ttbin` files with the Swabian Time Tagger API (`FileReader`). That API is built only for
Linux and Windows, so on macOS the decode runs in a Linux container. The PyPI wheel
`Swabian-TimeTagger==2.22.6` (manylinux, x86_64 and aarch64) reads the files written by Time Tagger 2.8.0.
It reports a few hundred events of type Error with channel -134217728 per file; they do not affect the decode.

## Setup on macOS (colima, no Docker Desktop)

Run from the repository root:

```sh
brew install colima docker
# On Apple Silicon this starts a native arm64 VM. Mount the repo so the container can see it.
colima start --cpu 6 --memory 16 --mount "$PWD:w"
docker build -t ppm-pnr-decode ppm_10ghz/src_decode/docker
```

On a Linux machine, skip colima and run only the `docker build` step.

## Run

Download the raw data first (`uv run python figshare_download.py --dataset ppm_20ghz_raw ppm_10ghz_raw`).
Outputs go to `<dataset>/inter/redecode/` by default, so the downloaded `inter/*.json` files are never
overwritten.

```sh
docker run --rm -v "$PWD":/work -w /work/ppm_20ghz/src_decode ppm-pnr-decode python redecode.py --db 48   # one file
docker run --rm -v "$PWD":/work -w /work/ppm_20ghz/src_decode ppm-pnr-decode python redecode.py --all      # 16-48 dB
docker run --rm -v "$PWD":/work -w /work/ppm_10ghz/src_decode ppm-pnr-decode python redecode.py --all      # 18-44 dB
```

- `redecode.py` extracts the `*.yml` files from `data/ground_truth.zip` into `data/ground_truth/` when that
  folder is missing.
- Per-file settings are in `FILE_LIST` in `JitRead_PPMSets.py` (10 GHz: slice adjustments) and
  `JitRead_PPMSets_20GHz.py` (20 GHz: number of events skipped at the start of each file).
- Each file takes about 20-60 s on an M-series Mac.

To evaluate a re-decode with the analysis scripts, copy (or move) the files from `inter/redecode/` to
`inter/` and run `scripts/run_all.sh`.

## Reproducibility

With this image the re-decode reproduces the published decode files:

- 20 GHz, 16-48 dB: every event of all five repetitions matches (`result`, `measured`,
  `gaussian_measured`, `true`, `tag`, `tag_x`, `tag_y`), as do the preamble points, slope-correction
  lookups and histograms. The 23 stored GMM fits agree to ~1e-8 relative (floating-point differences
  from the original Windows / Python 3.11 run).
- 10.75 GHz, 18-40 dB and 44 dB: event for event. 42 dB: the published file is the corrected re-decode
  (decode window moved out of the AWG sequence transition, `CustomFileTweaks(right_slice_adjustment=-150)`).
  With the old setting (`left=-20, right=-150`) the original faulty file is reproduced exactly.

scikit-learn is pinned to 1.2.2 because its `GaussianMixture` fit with `random_state=42` matches the original
decode. Versions 1.3 and later converge to slightly different optima, which moves the GMM error counts by a
few percent at some attenuations. Python 3.11 because scikit-learn 1.2.2 has no 3.12 wheels.
