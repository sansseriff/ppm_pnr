"""Non-interactive driver for JitRead_PPMSets.run_analysis.

Decodes selected attenuation files and writes decode_10GHz{dB}.json into an
output directory (default ../inter/redecode, so the original inter/*.json files
are never overwritten). Run from src_decode/, e.g. inside the docker image:

    python redecode.py --db 42 44
    python redecode.py --all --out ../inter/redecode
"""

import argparse
import os
import time
import zipfile

import orjson

import JitRead_PPMSets as jr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MEASURED = os.path.join(REPO, "data", "measured") + os.sep
GT_DIR = os.path.join(REPO, "data", "ground_truth")
GT_ZIP = os.path.join(REPO, "data", "ground_truth.zip")


def ensure_ground_truth():
    """Extract the yml part of ground_truth.zip (the decoder reads only yml)."""
    if os.path.isdir(GT_DIR) and any(f.endswith(".yml") for f in os.listdir(GT_DIR)):
        return
    print("extracting ground-truth yml files from", GT_ZIP)
    with zipfile.ZipFile(GT_ZIP) as z:
        for name in z.namelist():
            if name.startswith("ground_truth/") and name.endswith(".yml"):
                z.extract(name, os.path.dirname(GT_DIR))


def db_of(fname):
    return fname.split("_")[-1][:-8]  # "42.0.1.ttbin" -> "42.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="*", type=float, default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "inter", "redecode"))
    ap.add_argument(
        "--default-tweaks",
        action="store_true",
        help="ignore the per-file slice adjustments (keep offset/offset_index)",
    )
    args = ap.parse_args()

    ensure_ground_truth()
    os.makedirs(args.out, exist_ok=True)
    jr.DEBUG = False

    for fname, tweaks in jr.FILE_LIST:
        stub = db_of(fname)
        if not args.all and float(stub) not in args.db:
            continue
        if args.default_tweaks:
            tweaks = jr.CustomFileTweaks(offset=tweaks.offset, offset_index=tweaks.offset_index)
        print(f"=== decoding {fname} with {tweaks}", flush=True)
        t0 = time.time()
        results, gm_data, hist_data, correction_data, set_data = jr.run_analysis(
            MEASURED, fname, GT_DIR + os.sep, tweaks.offset, inter_path=args.out, cft=tweaks
        )
        out = jr.Out(results, gm_data, hist_data, correction_data, set_data)
        blob = orjson.dumps(
            out, default=lambda o: o.__dict__(), option=orjson.OPT_SERIALIZE_NUMPY
        )
        dest = os.path.join(args.out, f"decode_10GHz{stub}.json")
        with open(dest, "wb") as f:
            f.write(blob)
        print(f"wrote {dest} ({time.time() - t0:.0f} s)", flush=True)
        jr.plt.close("all")


if __name__ == "__main__":
    main()
