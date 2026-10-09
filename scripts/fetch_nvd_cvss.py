#!/usr/bin/env python3
"""
Fetch CVSS v3.1 vectors from the NVD API for every CVE in xamxte/cve-to-cwe.

Writes a JSON cache file (data/nvd_cache/cvss_vectors.json) mapping CVE-ID → CVSS vector.
Resumes from where it left off if interrupted.

Usage:
    python scripts/fetch_nvd_cvss.py --api-key YOUR_KEY
    python scripts/fetch_nvd_cvss.py --api-key YOUR_KEY --limit 100   # quick test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "data" / "nvd_cache" / "cvss_vectors.json"
NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key", required=True, help="NVD API key (free: nvd.nist.gov/developers)")
    p.add_argument("--limit", type=int, default=None, help="Only fetch first N CVEs (for testing)")
    p.add_argument("--batch-pause", type=float, default=0.7,
                   help="Seconds between requests. With API key: 50 req/30s → 0.6s safe. Default 0.7s.")
    return p.parse_args()


def load_cache() -> dict[str, str | None]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=0))


def fetch_cvss(cve_id: str, api_key: str) -> str | None:
    """Fetch CVSS v3.1 vector string for a single CVE. Returns None if unavailable."""
    try:
        resp = requests.get(NVD_API, params={"cveId": cve_id}, headers={"apiKey": api_key}, timeout=30)
        if resp.status_code == 403:
            print(f"  Rate limited, sleeping 30s...")
            time.sleep(30)
            resp = requests.get(NVD_API, params={"cveId": cve_id}, headers={"apiKey": api_key}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return None
        metrics = vulns[0]["cve"].get("metrics", {})
        for key in ("cvssMetricV31", "cvssMetricV30"):
            if key in metrics:
                return metrics[key][0]["cvssData"]["vectorString"]
        return None
    except Exception as e:
        print(f"  Error fetching {cve_id}: {e}")
        return None


def main() -> None:
    args = _parse()

    print("Loading xamxte/cve-to-cwe dataset...")
    ds = load_dataset("xamxte/cve-to-cwe", split="train")
    cve_ids = list(dict.fromkeys(ds["cve_id"]))  # unique, preserving order
    if args.limit:
        cve_ids = cve_ids[:args.limit]
    print(f"Total unique CVE IDs: {len(cve_ids)}")

    cache = load_cache()
    already = sum(1 for c in cve_ids if c in cache)
    print(f"Already cached: {already} | Remaining: {len(cve_ids) - already}")

    fetched, errors = 0, 0
    save_every = 500
    t0 = time.time()

    for i, cve_id in enumerate(cve_ids):
        if cve_id in cache:
            continue

        vector = fetch_cvss(cve_id, args.api_key)
        cache[cve_id] = vector
        fetched += 1
        if vector is None:
            errors += 1

        if fetched % 100 == 0:
            elapsed = (time.time() - t0) / 60
            rate = fetched / max(elapsed, 0.01)
            remaining = (len(cve_ids) - already - fetched) / max(rate, 0.01)
            print(f"  [{fetched}/{len(cve_ids) - already}] {cve_id}: {vector or 'NO CVSS'} "
                  f"| {elapsed:.1f}min elapsed, ~{remaining:.0f}min remaining")

        if fetched % save_every == 0:
            save_cache(cache)

        time.sleep(args.batch_pause)

    save_cache(cache)

    has_cvss = sum(1 for v in cache.values() if v is not None)
    print(f"\nDone! Fetched {fetched} new entries ({errors} without CVSS)")
    print(f"Cache total: {len(cache)} CVEs | {has_cvss} with CVSS ({has_cvss/len(cache)*100:.1f}%)")
    print(f"Saved to {CACHE_FILE}")


if __name__ == "__main__":
    main()
