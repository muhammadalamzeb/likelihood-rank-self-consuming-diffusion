#!/usr/bin/env python3
"""Upload manuscript PDF to Zenodo and publish (requires ZENODO_TOKEN)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ZDIR = ROOT / "paper" / "zenodo"
PDF = ZDIR / "Alamzeb_2026_likelihood_rank_self_consuming_diffusion.pdf"
META = json.loads((ZDIR / "metadata.json").read_text(encoding="utf-8"))
API = os.environ.get("ZENODO_API", "https://zenodo.org/api")


def req(method: str, url: str, token: str, data: bytes | None = None, content_type: str | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {e.code} {url}\n{err}") from e


def main() -> None:
    token = os.environ.get("ZENODO_TOKEN") or os.environ.get("ZENODO_ACCESS_TOKEN")
    if not token:
        print("Set ZENODO_TOKEN (Zenodo personal access token) and re-run.", file=sys.stderr)
        print("Create at: https://zenodo.org/account/settings/applications/tokens/new/", file=sys.stderr)
        raise SystemExit(2)
    if not PDF.exists():
        raise SystemExit(f"Missing PDF: {PDF}")

    print("Creating deposition...")
    _, dep = req("POST", f"{API}/deposit/depositions", token, data=b"{}", content_type="application/json")
    depo_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"  id={depo_id}")

    print(f"Uploading {PDF.name}...")
    pdf_bytes = PDF.read_bytes()
    put = urllib.request.Request(
        f"{bucket}/{PDF.name}",
        data=pdf_bytes,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
        method="PUT",
    )
    with urllib.request.urlopen(put, timeout=300) as resp:
        print(f"  upload status {resp.status}")

    src = ZDIR / "arxiv_source.zip"
    if src.exists():
        print(f"Uploading {src.name}...")
        put2 = urllib.request.Request(
            f"{bucket}/{src.name}",
            data=src.read_bytes(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
            },
            method="PUT",
        )
        with urllib.request.urlopen(put2, timeout=300) as resp:
            print(f"  upload status {resp.status}")

    print("Updating metadata...")
    payload = json.dumps(META).encode("utf-8")
    req(
        "PUT",
        f"{API}/deposit/depositions/{depo_id}",
        token,
        data=payload,
        content_type="application/json",
    )

    print("Publishing...")
    _, pub = req("POST", f"{API}/deposit/depositions/{depo_id}/actions/publish", token)
    doi = pub.get("doi") or pub.get("metadata", {}).get("doi")
    html = pub.get("links", {}).get("html")
    print("PUBLISHED")
    print(f"DOI: {doi}")
    print(f"URL: {html}")
    out = ZDIR / "PUBLISHED.txt"
    out.write_text(f"doi: {doi}\nurl: {html}\ndeposition_id: {depo_id}\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
