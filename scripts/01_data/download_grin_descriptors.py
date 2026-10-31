"""Download GRIN-Global soybean (crop 51) descriptor tables by direct connection.

The descriptor page exposes the observation table only through an ASP.NET
postback (button ``ctl00$MainContent$btnDownload``).  Files are written to
``down_data/GRIN_soybean_descriptors/`` as ``GRIN_crop51_<id>_<CODE>.csv``;
existing files are never overwritten and every new file is appended to
``MANIFEST.tsv`` (file, URL, date, sha256).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://npgsweb.ars-grin.gov/gringlobal/descriptordetail?id={}"
OUT = Path("/home/user/zhangzhishuai/myhermes/down_data/GRIN_soybean_descriptors")


def _hidden(html: str, name: str) -> str:
    m = re.search(r'name="%s"[^>]*value="([^"]*)"' % re.escape(name), html)
    return m.group(1) if m else ""


def fetch(descriptor_id: int, session: requests.Session) -> tuple[str, bytes]:
    url = BASE.format(descriptor_id)
    page = session.get(url, timeout=120)
    page.raise_for_status()
    html = page.text
    label = _hidden(html, "ctl00$MainContent$hf1")
    m = re.search(r"\(([^()]+)\)\s*$", label)
    if not m:
        raise RuntimeError(f"cannot parse descriptor code from {label!r}")
    code = re.sub(r"[^A-Za-z0-9_]+", "_", m.group(1)).strip("_").upper()
    form = {
        "__EVENTTARGET": "ctl00$MainContent$btnDownload",
        "__EVENTARGUMENT": "",
        "__VIEWSTATE": _hidden(html, "__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": _hidden(html, "__VIEWSTATEGENERATOR"),
        "__VIEWSTATEENCRYPTED": _hidden(html, "__VIEWSTATEENCRYPTED"),
        "__EVENTVALIDATION": _hidden(html, "__EVENTVALIDATION"),
        "ctl00$MainContent$hf1": label,
    }
    resp = session.post(url, data=form, timeout=600)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if "csv" not in ctype and not resp.content.startswith(b"Accession evaluated"):
        raise RuntimeError(f"descriptor {descriptor_id}: unexpected content type {ctype!r}")
    return code, resp.content


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", type=int, nargs="+")
    args = ap.parse_args(argv)
    session = requests.Session()
    session.trust_env = False  # never use the local proxy for data downloads
    session.headers["User-Agent"] = "Mozilla/5.0 (research data download)"
    manifest = OUT / "MANIFEST.tsv"
    for did in args.ids:
        code, content = fetch(did, session)
        path = OUT / f"GRIN_crop51_{did}_{code}.csv"
        if path.exists():
            print(f"skip existing {path.name}")
            continue
        path.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with manifest.open("a") as fh:
            fh.write(f"{path.name}\t{BASE.format(did)} (Download postback)\t{stamp} (direct, no proxy)\t{sha}\n")
        print(f"wrote {path.name} ({len(content)} bytes)")
        time.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
