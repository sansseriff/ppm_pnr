"""Download the datasets of this repository from figshare and unpack them where the scripts expect them.

    uv run python figshare_download.py --list                          # what exists, what is already here
    uv run python figshare_download.py --dataset ppm_20ghz_decoded waveform
    uv run python figshare_download.py --all                           # everything (~3.4 GB of zips)
    uv run python figshare_download.py                                 # interactive menu

Datasets (see figshare_metadata.yaml):
    ppm_20ghz_decoded   decoded 20 GHz data: everything the 20 GHz figures need   (27 MB zip)
    ppm_10ghz_decoded   decoded 10.75 GHz data (supplemental figures)              (114 MB zip)
    waveform            detector waveform scan (Fig. 2)                            (0.1 MB zip)
    ppm_20ghz_raw       raw 20 GHz time tags + AWG ground truth (only for re-decoding)   (2.0 GB zip)
    ppm_10ghz_raw       raw 10.75 GHz time tags + AWG ground truth                       (1.3 GB zip)

Every zip holds paths relative to the repository root (e.g. ppm_20ghz/inter/decode_20GHz24.0.json) and is
unpacked into the repository root. A dataset whose files are all present is skipped (use --force to
download it again). The zip is checked against the md5 that figshare computed on upload, and after
unpacking every file is checked against the dataset's MANIFEST_<key>.txt.

Before the figshare items are published they are private: set FIGSHARE_TOKEN to a personal token of an
account with access (figshare.com -> Applications and personal tokens). After publication no token is needed.
"""

import argparse
import hashlib
import os
import shutil
import sys
import zipfile

import requests
import yaml

API = "https://api.figshare.com/v2"
REPO = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(REPO, "figshare_metadata.yaml")
CHUNK = 1 << 20


def load_meta():
    with open(META) as f:
        return yaml.safe_load(f)


def human(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} PB"


def session():
    s = requests.Session()
    tok = os.environ.get("FIGSHARE_TOKEN")
    if tok:
        s.headers["Authorization"] = f"token {tok}"
    return s


def article_files(s, article_id):
    """file list of an item: public API first, then the private (account) API if a token is set."""
    r = s.get(f"{API}/articles/{article_id}/files", timeout=60)
    if r.status_code == 200:
        return r.json()
    if "Authorization" in s.headers:
        r = s.get(f"{API}/account/articles/{article_id}/files", timeout=60)
        if r.status_code == 200:
            return r.json()
        sys.exit(f"figshare item {article_id}: HTTP {r.status_code} with FIGSHARE_TOKEN: {r.text[:200]}")
    sys.exit(f"figshare item {article_id} is not public (HTTP {r.status_code}). If it has not been published "
             f"yet, set FIGSHARE_TOKEN to a personal token with access to it.")


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK * 4), b""):
            h.update(b)
    return h.hexdigest()


def present(d):
    root = os.path.join(REPO, d.get("extract_to", "."))
    files = list(d.get("files") or []) + ([d["manifest"]] if d.get("manifest") else [])
    missing = [f for f in files if not os.path.exists(os.path.join(root, f))]
    return len(files) > 0 and not missing, missing


def download(s, url, dest, size):
    tmp = dest + ".part"
    with s.get(url, stream=True, timeout=120) as r:
        if r.status_code != 200:
            sys.exit(f"download {url}: HTTP {r.status_code}")
        done, next_report = 0, 0.0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if size and done / size >= next_report:
                    print(f"\r  {human(done)} / {human(size)} ({100 * done / size:.0f}%)", end="", flush=True)
                    next_report += 0.02
    print()
    os.replace(tmp, dest)


def safe_extract(zpath, root, allowed):
    """unpack only the expected members, refusing anything that would land outside root."""
    root = os.path.realpath(root)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            target = os.path.realpath(os.path.join(root, info.filename))
            if not target.startswith(root + os.sep) or (allowed and info.filename not in allowed):
                sys.exit(f"{zpath}: unexpected member {info.filename!r}, not extracting")
        for info in z.infolist():
            if info.is_dir():
                continue
            target = os.path.join(root, info.filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)


def check_manifest(root, manifest):
    bad = []
    with open(os.path.join(root, manifest)) as f:
        for line in f:
            md5, size, path = line.split(maxsplit=2)
            path = path.strip()
            p = os.path.join(root, path)
            if not os.path.exists(p) or os.path.getsize(p) != int(size) or md5sum(p) != md5:
                bad.append(path)
    return bad


def fetch(s, key, d, args):
    root = os.path.join(REPO, d.get("extract_to", "."))
    ok, _ = present(d)
    if ok and not args.force:
        print(f"{key}: already present, skipping (--force to download again)")
        if args.verify:
            bad = check_manifest(root, d["manifest"])
            print(f"{key}: manifest check {'OK' if not bad else 'FAILED for ' + ', '.join(bad)}")
        return
    if not d.get("article_id"):
        sys.exit(f"{key}: no figshare article id in figshare_metadata.yaml")
    files = article_files(s, d["article_id"])
    fm = next((f for f in files if f["name"] == d["zip"]), None)
    if fm is None:
        sys.exit(f"{key}: {d['zip']} not found in figshare item {d['article_id']} ({[f['name'] for f in files]})")
    os.makedirs(args.download_dir, exist_ok=True)
    zpath = os.path.join(args.download_dir, fm["name"])
    expected_md5 = fm.get("computed_md5") or d.get("zip_md5")
    if os.path.exists(zpath) and md5sum(zpath) == expected_md5:
        print(f"{key}: using already downloaded {zpath}")
    else:
        print(f"{key}: downloading {fm['name']} ({human(fm['size'])}) from figshare item {d['article_id']}")
        download(s, fm["download_url"], zpath, fm["size"])
        got = md5sum(zpath)
        if got != expected_md5:
            sys.exit(f"{key}: md5 mismatch for {zpath}: got {got}, figshare says {expected_md5}")
        print(f"{key}: md5 OK ({got})")
    allowed = set(d.get("files") or []) | ({d["manifest"]} if d.get("manifest") else set())
    print(f"{key}: unpacking into {os.path.relpath(root, os.getcwd()) or '.'}")
    safe_extract(zpath, root, allowed)
    bad = check_manifest(root, d["manifest"])
    if bad:
        sys.exit(f"{key}: files differ from {d['manifest']}: {bad}")
    print(f"{key}: {len(allowed) - 1} files OK")
    if not args.keep_zip:
        os.remove(zpath)


def print_list(meta):
    print(f"{'dataset':20s} {'zip':>9s} {'unpacked':>9s}  {'here?':6s} {'doi':32s} description")
    for key, d in meta["datasets"].items():
        ok, _ = present(d)
        print(f"{key:20s} {human(d.get('zip_size', 0)):>9s} {human(d.get('unpacked_size', 0)):>9s}  "
              f"{'yes' if ok else 'no':6s} {str(d.get('doi') or '-'):32s} {d.get('summary', '')}")


def menu(meta):
    keys = list(meta["datasets"])
    print_list(meta)
    print()
    for i, k in enumerate(keys, 1):
        print(f"  {i}) {k}")
    ans = input("datasets to download (numbers or names separated by spaces, 'all', ENTER to quit): ").split()
    if not ans:
        return []
    if ans == ["all"]:
        return keys
    out = []
    for a in ans:
        if a.isdigit() and 1 <= int(a) <= len(keys):
            out.append(keys[int(a) - 1])
        elif a in keys:
            out.append(a)
        else:
            sys.exit(f"unknown choice {a!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list datasets and whether they are present")
    ap.add_argument("--dataset", nargs="+", metavar="KEY", help="datasets to download")
    ap.add_argument("--all", action="store_true", help="download every dataset")
    ap.add_argument("--force", action="store_true", help="download even if the files are present")
    ap.add_argument("--verify", action="store_true", help="md5-check files of datasets that are skipped")
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip files")
    ap.add_argument("--download-dir", default=os.path.join(REPO, "downloads"), help="where zips are saved")
    args = ap.parse_args()
    meta = load_meta()
    if args.list:
        print_list(meta)
        return
    keys = list(meta["datasets"]) if args.all else args.dataset or menu(meta)
    unknown = [k for k in keys if k not in meta["datasets"]]
    if unknown:
        sys.exit(f"unknown dataset(s) {unknown}; choose from {list(meta['datasets'])}")
    s = session()
    for k in keys:
        fetch(s, k, meta["datasets"][k], args)
    if os.path.isdir(args.download_dir) and not os.listdir(args.download_dir):
        os.rmdir(args.download_dir)


if __name__ == "__main__":
    main()
