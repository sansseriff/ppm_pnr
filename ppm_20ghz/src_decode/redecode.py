"""Non-interactive driver for JitRead_PPMSets_20GHz.run_analysis (the `full_decode` branch of its __main__).

Decodes selected attenuation files and writes decode_20GHz{dB}.json into an output directory (default
../inter/redecode, so the downloaded inter/*.json files are never overwritten). Needs the Swabian Time Tagger
API, i.e. run it inside the Linux container (see ../../ppm_10ghz/src_decode/docker/README.md):

    python redecode.py --db 48
    python redecode.py --all --out ../inter/redecode
"""

import argparse
import os
import time
import zipfile

import orjson

import JitRead_PPMSets_20GHz as jr

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.dirname(HERE)
MEASURED = os.path.join(DATASET, "data", "measured", "Sept10_20GHz") + os.sep
GT_DIR = os.path.join(DATASET, "data", "ground_truth")
GT_ZIP = os.path.join(DATASET, "data", "ground_truth.zip")


def ensure_ground_truth():
    """Extract the yml part of ground_truth.zip (the decoder reads only yml, and import_ground_truth takes the
    date stamp from the first file it lists, so the folder must not contain anything else)."""
    if os.path.isdir(GT_DIR) and any(f.endswith(".yml") for f in os.listdir(GT_DIR)):
        return
    print("extracting ground-truth yml files from", GT_ZIP)
    with zipfile.ZipFile(GT_ZIP) as z:
        for name in z.namelist():
            if name.startswith("ground_truth/") and name.endswith(".yml"):
                z.extract(name, os.path.dirname(GT_DIR))


def db_of(fname):
    return fname.split("_")[-1][:-8]  # "48.0.1.ttbin" -> "48.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="*", type=float, default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.join(DATASET, "inter", "redecode"))
    args = ap.parse_args()

    ensure_ground_truth()
    os.makedirs(args.out, exist_ok=True)
    jr.DEBUG = False

    for fname, start_index in jr.FILE_LIST:
        stub = db_of(fname)
        if not args.all and float(stub) not in args.db:
            continue
        print(f"=== decoding {fname}, skipping the first {start_index} events", flush=True)
        t0 = time.time()
        results, gm_data, hist_data, correction_data = jr.run_analysis(
            MEASURED, fname, GT_DIR + os.sep, start_index, inter_path=args.out
        )
        out = jr.Out(results, gm_data, hist_data, correction_data)
        blob = orjson.dumps(out, default=lambda o: o.__dict__(), option=orjson.OPT_SERIALIZE_NUMPY)
        dest = os.path.join(args.out, f"decode_20GHz{stub}.json")
        with open(dest, "wb") as f:
            f.write(blob)
        print(f"wrote {dest} ({time.time() - t0:.0f} s)", flush=True)
        jr.plt.close("all")


if __name__ == "__main__":
    main()
