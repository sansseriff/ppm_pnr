"""Maintainer tool: create the figshare items for this repository's datasets and upload the zips.

Everything here works on PRIVATE (draft) items. This script never publishes anything: publishing is done
by the author on figshare.com (or with the API) once the manuscript is accepted.

Token: read from the FIGSHARE_TOKEN environment variable, or from ~/.config/figshare/token.
It is never printed or written anywhere.

    uv run --group tools python tools/figshare_upload.py status
    uv run --group tools python tools/figshare_upload.py create                 # project + one item per dataset, reserve DOIs
    uv run --group tools python tools/figshare_upload.py update-metadata        # push title/description/keywords again
    uv run --group tools python tools/figshare_upload.py upload --zips DIR [KEY ...]   # resumable, chunked
    uv run --group tools python tools/figshare_upload.py verify [KEY ...]       # compare figshare's computed_md5

Item ids, DOIs and file ids are written back into figshare_metadata.yaml.

Upload protocol (https://docs.figshare.com/#upload_files): POST /account/articles/{id}/files with name, size
and md5 -> GET the file's upload_url -> GET the list of parts -> PUT every part that is not COMPLETE ->
POST /account/articles/{id}/files/{file_id} to complete. Interrupted uploads are resumed by re-running the
command: parts already marked COMPLETE on the server are skipped.
"""

import argparse
import concurrent.futures as cf
import hashlib
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml_rt  # noqa: E402

API = "https://api.figshare.com/v2"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(REPO, "figshare_metadata.yaml")
LICENSES = {"CC BY 4.0": 1, "CC0": 2, "MIT": 3}


def token():
    t = os.environ.get("FIGSHARE_TOKEN")
    if not t:
        p = os.path.expanduser("~/.config/figshare/token")
        if os.path.exists(p):
            t = open(p).read().strip()
    if not t:
        sys.exit("no figshare token: set FIGSHARE_TOKEN or write it to ~/.config/figshare/token")
    return t


class Client:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"token {token()}"

    def call(self, method, url, retries=6, **kw):
        if not url.startswith("http"):
            url = API + url
        for attempt in range(retries):
            try:
                r = self.s.request(method, url, timeout=kw.pop("timeout", 120), **kw)
            except requests.RequestException as e:
                err = repr(e)
            else:
                if r.status_code < 500 and r.status_code != 429:
                    if r.status_code >= 400:
                        raise RuntimeError(f"{method} {url} -> {r.status_code}: {r.text[:500]}")
                    return r
                err = f"{r.status_code} {r.text[:200]}"
            wait = min(60, 2 ** attempt)
            print(f"  {method} {url}: {err}; retry in {wait} s", flush=True)
            time.sleep(wait)
        raise RuntimeError(f"{method} {url} failed after {retries} attempts")

    def json(self, method, url, **kw):
        r = self.call(method, url, **kw)
        return r.json() if r.content else None


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def article_body(meta, key):
    c, d = meta["common"], meta["datasets"][key]
    return dict(
        title=str(d["title"]),
        description=" ".join(str(d["description"]).split()),
        keywords=[str(k) for k in c["keywords"]],
        categories=[int(x) for x in c["categories"]],
        license=LICENSES[str(c["license"])],
        defined_type=str(c["defined_type"]),
    )


def cmd_status(cl, meta, args):
    acct = cl.json("GET", "/account")
    print(f"private quota used: {acct['used_quota_private'] / 1e9:.2f} GB of {acct['quota'] / 1e9:.1f} GB")
    pid = meta["project"].get("project_id")
    print(f"project: {pid}")
    for key, d in meta["datasets"].items():
        aid = d.get("article_id")
        if not aid:
            print(f"{key:20s} (no item yet)")
            continue
        a = cl.json("GET", f"/account/articles/{aid}")
        files = cl.json("GET", f"/account/articles/{aid}/files")
        fs = ", ".join(f"{f['name']} {f['size'] / 1e6:.1f} MB [{f['status']}]" for f in files) or "no files"
        print(f"{key:20s} id={aid} status={a['status']} doi={a.get('doi') or '-'}  {fs}")


def find_project(cl, title):
    for p in cl.json("GET", "/account/projects", params=dict(page_size=1000)):
        if p["title"] == title:
            return p["id"]
    return None


def cmd_create(cl, meta, args):
    proj = meta["project"]
    pid = proj.get("project_id") or find_project(cl, str(proj["title"]))
    if not pid:
        r = cl.json("POST", "/account/projects",
                    json=dict(title=str(proj["title"]), description=" ".join(str(proj["description"]).split())))
        pid = r["entity_id"]
        print(f"created project {pid}")
    proj["project_id"] = pid
    proj["url"] = f"https://figshare.com/account/projects/{pid}"
    yaml_rt.dump(meta, META)

    existing = {a["title"]: a["id"] for a in cl.json("GET", f"/account/projects/{pid}/articles",
                                                      params=dict(page_size=1000))}
    for key in args.keys or list(meta["datasets"]):
        d = meta["datasets"][key]
        body = article_body(meta, key)
        aid = d.get("article_id") or existing.get(body["title"])
        if not aid:
            r = cl.json("POST", f"/account/projects/{pid}/articles", json=body)
            aid = r["entity_id"]
            print(f"{key}: created item {aid}")
        else:
            cl.call("PUT", f"/account/articles/{aid}", json=body)
            print(f"{key}: item {aid} exists, metadata updated")
        d["article_id"] = aid
        a = cl.json("GET", f"/account/articles/{aid}")
        doi = a.get("doi")
        if not doi and not args.no_doi:
            doi = cl.json("POST", f"/account/articles/{aid}/reserve_doi")["doi"]
            print(f"{key}: reserved DOI {doi}")
        d["doi"] = doi or None
        d["url_private_html"] = f"https://figshare.com/account/articles/{aid}"
        d["url_public_html"] = f"https://doi.org/{doi}" if doi else None
        yaml_rt.dump(meta, META)


def cmd_update_metadata(cl, meta, args):
    for key in args.keys or list(meta["datasets"]):
        aid = meta["datasets"][key]["article_id"]
        cl.call("PUT", f"/account/articles/{aid}", json=article_body(meta, key))
        print(f"{key}: metadata updated on item {aid}")


def upload_file(cl, aid, path, md5, workers=4):
    name, size = os.path.basename(path), os.path.getsize(path)
    files = {f["name"]: f for f in cl.json("GET", f"/account/articles/{aid}/files")}
    f = files.get(name)
    if f is not None and (f["size"] != size or (f.get("supplied_md5") and f["supplied_md5"] != md5)):
        print(f"  {name}: a different version is on figshare (file {f['id']}), deleting it", flush=True)
        cl.call("DELETE", f"/account/articles/{aid}/files/{f['id']}")
        f = None
    if f is not None and f["status"] == "available":
        print(f"  {name}: already uploaded (file {f['id']})", flush=True)
        return f["id"]
    if f is None:
        loc = cl.json("POST", f"/account/articles/{aid}/files", json=dict(name=name, size=size, md5=md5))["location"]
        f = cl.json("GET", loc)
        print(f"  {name}: new upload, file {f['id']}", flush=True)
    else:
        f = cl.json("GET", f"/account/articles/{aid}/files/{f['id']}")
        print(f"  {name}: resuming upload of file {f['id']}", flush=True)
    up = cl.json("GET", f["upload_url"])
    todo = [p for p in up["parts"] if p["status"] != "COMPLETE"]
    print(f"  {name}: {len(up['parts'])} parts, {len(todo)} to send", flush=True)

    def put(part):
        with open(path, "rb") as fh:
            fh.seek(part["startOffset"])
            data = fh.read(part["endOffset"] - part["startOffset"] + 1)
        cl.call("PUT", f"{f['upload_url']}/{part['partNo']}", data=data, timeout=600)
        return len(data)

    sent, t0 = 0, time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, n in enumerate(ex.map(put, todo)):
            sent += n
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                rate = sent / max(time.time() - t0, 1e-6) / 1e6
                print(f"  {name}: {i + 1}/{len(todo)} parts, {sent / 1e6:.0f} MB, {rate:.1f} MB/s", flush=True)
    cl.call("POST", f"/account/articles/{aid}/files/{f['id']}")
    print(f"  {name}: upload completed", flush=True)
    return f["id"]


def cmd_upload(cl, meta, args):
    for key in args.keys or list(meta["datasets"]):
        d = meta["datasets"][key]
        path = os.path.join(args.zips, str(d["zip"]))
        md5 = str(d.get("zip_md5") or md5sum(path))
        if md5sum(path) != md5:
            sys.exit(f"{path}: md5 differs from figshare_metadata.yaml; re-run package_datasets.py")
        print(f"{key}: uploading {path} ({os.path.getsize(path) / 1e6:.1f} MB) to item {d['article_id']}", flush=True)
        d["file_id"] = upload_file(cl, d["article_id"], path, md5, workers=args.workers)
        meta = yaml_rt.load(META)  # another process may have edited the file meanwhile
        meta["datasets"][key]["file_id"] = d["file_id"]
        yaml_rt.dump(meta, META)


def cmd_verify(cl, meta, args):
    ok = True
    for key in args.keys or list(meta["datasets"]):
        d = meta["datasets"][key]
        files = cl.json("GET", f"/account/articles/{d['article_id']}/files")
        f = next((f for f in files if f["name"] == d["zip"]), None)
        if f is None:
            print(f"{key}: {d['zip']} not on figshare")
            ok = False
            continue
        good = f["status"] == "available" and f.get("computed_md5") == d["zip_md5"] and f["size"] == d["zip_size"]
        ok &= good
        print(f"{key}: {'OK ' if good else 'BAD'} status={f['status']} size={f['size']} "
              f"computed_md5={f.get('computed_md5')} expected={d['zip_md5']}")
        extra = [x["name"] for x in files if x["name"] != d["zip"]]
        if extra:
            print(f"  other files on the item: {extra}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("create")
    p.add_argument("keys", nargs="*")
    p.add_argument("--no-doi", action="store_true", help="do not reserve DOIs")
    p = sub.add_parser("update-metadata")
    p.add_argument("keys", nargs="*")
    p = sub.add_parser("upload")
    p.add_argument("keys", nargs="*")
    p.add_argument("--zips", required=True, help="folder with the zips from package_datasets.py")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("verify")
    p.add_argument("keys", nargs="*")
    args = ap.parse_args()
    meta = yaml_rt.load(META)
    cl = Client()
    dict(status=cmd_status, create=cmd_create, upload=cmd_upload, verify=cmd_verify,
         **{"update-metadata": cmd_update_metadata})[args.cmd](cl, meta, args)


if __name__ == "__main__":
    main()
