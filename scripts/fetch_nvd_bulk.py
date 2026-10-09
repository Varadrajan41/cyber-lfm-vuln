#!/usr/bin/env python3
"""Bulk-fetch CVSS v3.1 vectors from NVD API v2.0 using pagination.

Instead of querying one CVE at a time (~45 hours for 234k CVEs),
this paginates through ALL NVD CVEs in pages of 2000, extracting
CVSS vectors only for CVE IDs in our training dataset.

~125 pages × ~3s each ≈ 6-8 minutes total.
"""

import json, time, argparse, os, sys
from pathlib import Path

import requests
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "nvd_cache" / "cvss_vectors.json"
NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.rename(CACHE)


def extract_cvss_vector(cve_obj):
    """Extract CVSS v3.1 (or v3.0 fallback) vector string."""
    metrics = cve_obj.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        for m in metrics.get(key, []):
            vec = m.get("cvssData", {}).get("vectorString")
            if vec:
                return vec
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("NVD_API_KEY"))
    p.add_argument("--page-size", type=int, default=2000)
    p.add_argument("--pause", type=float, default=0.8,
                   help="Seconds between requests (0.6 is safe with API key)")
    args = p.parse_args()

    print("Loading target CVE IDs from xamxte/cve-to-cwe...")
    ds = load_dataset("xamxte/cve-to-cwe", split="train")
    target_cves = set(ds["cve_id"])
    print(f"  {len(target_cves):,} unique CVEs needed")

    cache = load_cache()
    already_cached = sum(1 for c in target_cves if c in cache)
    already_with_cvss = sum(1 for c in target_cves if cache.get(c))
    print(f"  {already_cached:,} already cached ({already_with_cvss:,} with CVSS)")
    remaining = len(target_cves) - already_cached
    print(f"  {remaining:,} remaining to look up\n")

    if remaining == 0:
        print("All target CVEs already in cache!")
        return

    headers = {}
    if args.api_key:
        headers["apiKey"] = args.api_key
        print("Using API key (higher rate limit)")
    else:
        print("⚠ No API key — will be slower. Set NVD_API_KEY env var.")

    # Get total count
    print("Querying NVD for total CVE count...")
    r = requests.get(NVD_API, params={"resultsPerPage": 1, "startIndex": 0},
                     headers=headers, timeout=60)
    r.raise_for_status()
    total_nvd = r.json()["totalResults"]
    pages = (total_nvd + args.page_size - 1) // args.page_size
    print(f"NVD has {total_nvd:,} CVEs → {pages} pages of {args.page_size}\n")

    t0 = time.time()
    new_matched = 0
    new_with_cvss = 0

    for page in range(pages):
        start_idx = page * args.page_size

        for attempt in range(3):
            try:
                resp = requests.get(
                    NVD_API,
                    params={"resultsPerPage": args.page_size,
                            "startIndex": start_idx},
                    headers=headers,
                    timeout=120,
                )
                if resp.status_code == 403:
                    wait = 35 if attempt == 0 else 60
                    print(f"  Rate limited (403), waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"  Request error: {e}, retry {attempt+1}/3...")
                time.sleep(10)
        else:
            print(f"  Skipping page {page} after 3 failures")
            continue

        data = resp.json()
        vulns = data.get("vulnerabilities", [])

        page_matched = 0
        for item in vulns:
            cve_obj = item.get("cve", {})
            cve_id = cve_obj.get("id")
            if not cve_id or cve_id not in target_cves or cve_id in cache:
                continue
            vector = extract_cvss_vector(cve_obj)
            cache[cve_id] = vector
            new_matched += 1
            if vector:
                new_with_cvss += 1
            page_matched += 1

        elapsed = time.time() - t0
        total_with_cvss = sum(1 for c in target_cves if cache.get(c))
        total_cached = sum(1 for c in target_cves if c in cache)
        eta = (elapsed / (page + 1)) * (pages - page - 1)

        print(f"  [{page+1:>3}/{pages}] "
              f"NVD {min(start_idx + args.page_size, total_nvd):>7,}/{total_nvd:,} | "
              f"+{page_matched} this page | "
              f"total: {total_cached:,}/{len(target_cves):,} cached, "
              f"{total_with_cvss:,} with CVSS | "
              f"ETA {eta/60:.1f}m")

        # Save every 10 pages
        if (page + 1) % 10 == 0:
            save_cache(cache)

        time.sleep(args.pause)

    save_cache(cache)

    total_cached = sum(1 for c in target_cves if c in cache)
    total_with_cvss = sum(1 for c in target_cves if cache.get(c))
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"  Cached:    {total_cached:,} / {len(target_cves):,}")
    print(f"  With CVSS: {total_with_cvss:,}")
    print(f"  No CVSS:   {total_cached - total_with_cvss:,}")
    print(f"  New this run: {new_matched} matched, {new_with_cvss} with CVSS")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
