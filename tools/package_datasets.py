"""Build the figshare zip archives from the author's working data folders.

Each dataset becomes one zip whose internal paths are the paths inside this repository, so that
`unzip <key>.zip -d <repo root>` (or figshare_download.py) puts every file where the scripts expect it.
A manifest (md5, size, path) is written next to the data inside each zip and printed to
<out>/<key>.files.json, which this script also uses to fill the `files` list in figshare_metadata.yaml.

This is a maintainer tool: users never need it. Example:

    uv run python tools/package_datasets.py --source-root ../final_data --out /tmp/ppm_pnr_zips
    uv run python tools/package_datasets.py --source-root ../final_data --out /tmp/ppm_pnr_zips --only waveform

(--source-root is only needed when building the zips, not with --metadata-only.)
"""

import argparse
import glob
import hashlib
import json
import os
import zipfile

import yaml_rt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# key -> list of (glob relative to --source-root, destination directory inside the repo), manifest dir
SPEC = {
    "ppm_20ghz_raw": dict(
        manifest_dir="ppm_20ghz/data",
        items=[
            # 16..50 dB; the decoder reads the *.1.ttbin files (the *.0.ttbin files are 2.7 kB headers).
            # The testing/ subfolder of the original data was not used for the paper and is not included.
            ("ppm_20ghz/data/measured/Sept10_20GHz/*.ttbin", "ppm_20ghz/data/measured/Sept10_20GHz"),
            ("ppm_20ghz/data/ground_truth.zip", "ppm_20ghz/data"),
        ],
    ),
    "ppm_20ghz_decoded": dict(
        manifest_dir="ppm_20ghz/inter",
        items=[
            ("ppm_20ghz/inter/decode_20GHz*.0.json", "ppm_20ghz/inter"),
            ("ppm_20ghz/inter/static_sim.json", "ppm_20ghz/inter"),
            ("ppm_20ghz/inter/pnr_refit.json", "ppm_20ghz/inter"),
            ("ppm_20ghz/inter/offset_correction_*.json", "ppm_20ghz/inter"),
            ("ppm_20ghz/inter/max_offsets.json", "ppm_20ghz/inter"),
        ],
    ),
    "ppm_10ghz_raw": dict(
        manifest_dir="ppm_10ghz/data",
        items=[
            ("ppm_10ghz/data/measured/*.ttbin", "ppm_10ghz/data/measured"),
            ("ppm_10ghz/data/ground_truth.zip", "ppm_10ghz/data"),
        ],
    ),
    "ppm_10ghz_decoded": dict(
        manifest_dir="ppm_10ghz/inter",
        items=[
            # decode_10GHz42.0.json is the corrected re-decode (decode-window fix, Oct 2026)
            ("ppm_10ghz/inter/decode_10GHz*.0.json", "ppm_10ghz/inter"),
            ("ppm_10ghz/inter/static_sim.json", "ppm_10ghz/inter"),
            ("ppm_10ghz/inter/pnr_refit.json", "ppm_10ghz/inter"),
        ],
    ),
    "waveform": dict(
        manifest_dir="waveform/data",
        items=[
            ("waveform/data/decent.npy", "waveform/data"),
            ("waveform/src/tigger_level_time_difference.json", "waveform/data"),
        ],
    ),
}


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def collect(source_root, key):
    files = []
    for pattern, dest in SPEC[key]["items"]:
        matches = sorted(glob.glob(os.path.join(source_root, pattern)))
        if not matches:
            raise FileNotFoundError(f"{key}: nothing matches {pattern}")
        for m in matches:
            files.append((m, f"{dest}/{os.path.basename(m)}"))
    return files


def build(source_root, key, out_dir):
    files = collect(source_root, key)
    manifest = []
    for src, arc in files:
        manifest.append(dict(path=arc, size=os.path.getsize(src), md5=md5sum(src)))
    manifest_name = f"{SPEC[key]['manifest_dir']}/MANIFEST_{key}.txt"
    manifest_txt = "".join(f"{m['md5']}  {m['size']:>12d}  {m['path']}\n" for m in manifest)

    zpath = os.path.join(out_dir, f"{key}.zip")
    tmp = zpath + ".part"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for src, arc in files:
            print(f"  {key}: adding {arc}", flush=True)
            z.write(src, arc)
        z.writestr(manifest_name, manifest_txt)
    os.replace(tmp, zpath)
    info = dict(
        key=key,
        zip=os.path.basename(zpath),
        zip_size=os.path.getsize(zpath),
        zip_md5=md5sum(zpath),
        unpacked_size=sum(m["size"] for m in manifest),
        manifest=manifest_name,
        files=manifest,
    )
    with open(os.path.join(out_dir, f"{key}.files.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"{key}: {len(files)} files, {info['unpacked_size'] / 1e6:.1f} MB -> {info['zip_size'] / 1e6:.1f} MB zip",
          flush=True)
    return info


def update_metadata(infos, meta_path):
    """write sizes and file lists into figshare_metadata.yaml (keeps all other fields and comments)."""
    meta = yaml_rt.load(meta_path)
    for info in infos:
        d = meta["datasets"][info["key"]]
        d["zip"] = info["zip"]
        d["zip_size"] = info["zip_size"]
        d["zip_md5"] = info["zip_md5"]
        d["unpacked_size"] = info["unpacked_size"]
        d["manifest"] = info["manifest"]
        d["files"] = [f["path"] for f in info["files"]]
    yaml_rt.dump(meta, meta_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", default=None, help="folder holding ppm_20ghz/, ppm_10ghz/, waveform/")
    ap.add_argument("--out", required=True, help="output folder for the zips")
    ap.add_argument("--only", nargs="*", choices=list(SPEC), default=list(SPEC))
    ap.add_argument("--metadata", default=os.path.join(REPO, "figshare_metadata.yaml"))
    ap.add_argument("--no-metadata", action="store_true", help="do not touch figshare_metadata.yaml")
    ap.add_argument("--metadata-only", action="store_true",
                    help="do not build zips, only copy <out>/<key>.files.json into figshare_metadata.yaml")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.metadata_only:
        infos = [json.load(open(os.path.join(args.out, f"{key}.files.json"))) for key in args.only]
    else:
        if not args.source_root:
            ap.error("--source-root is required unless --metadata-only")
        infos = [build(args.source_root, key, args.out) for key in args.only]
    if not args.no_metadata:
        update_metadata(infos, args.metadata)


if __name__ == "__main__":
    main()
