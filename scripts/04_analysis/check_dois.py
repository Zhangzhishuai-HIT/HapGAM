"""Verify DOIs against Crossref (direct connection) and print citation metadata."""
import json
import os
import sys
import time

import requests

dois = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
s = requests.Session()
s.trust_env = False
mailto = os.environ.get("CROSSREF_MAILTO")  # contact address for the Crossref polite pool (optional)
s.headers["User-Agent"] = "HapGAM-reference-check" + (f" (mailto:{mailto})" if mailto else "")
out = []
for d in dois:
    try:
        r = s.get(f"https://api.crossref.org/works/{d}", timeout=60)
        if r.status_code != 200:
            print(f"FAIL {d} HTTP {r.status_code}", flush=True)
            out.append({"doi": d, "ok": False})
            continue
        m = r.json()["message"]
        au = m.get("author", [])
        names = [f"{a.get('family', '')} {''.join(x[0] for x in a.get('given', '').replace('-', ' ').split())}" for a in au[:3]]
        etal = " et al." if len(au) > 3 else ""
        year = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        cite = (f"{', '.join(names)}{etal} ({year}) {m.get('title', [''])[0]}. {(m.get('container-title') or [''])[0]} "
                f"{m.get('volume', '')}:{m.get('page', m.get('article-number', ''))}. doi:{d}")
        print("OK  ", cite, flush=True)
        out.append({"doi": d, "ok": True, "citation": cite, "year": year})
    except Exception as e:  # noqa: BLE001
        print(f"ERR {d} {e}", flush=True)
        out.append({"doi": d, "ok": False})
    time.sleep(0.5)
json.dump(out, open(sys.argv[2], "w"), indent=1)
