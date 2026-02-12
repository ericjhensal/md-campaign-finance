"""Download CSV data directly from the Maryland CRIS API.

Usage:
    python download_data.py                  # Download current cycle (2023-2026)
    python download_data.py --year 2025      # Download specific year
    python download_data.py --type contrib   # Download only contributions

The CRIS download API uses these transaction codes:
    Contributions: TCOC (full cycle) / TCON (single year)
    Expenditures:  TEXC (full cycle) / TEXP (single year)
    Committees:    TCMC (full cycle) / TCOM (single year)
"""
import os
import sys
import argparse
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CRIS_API_URL, RAW_DIR

DOWNLOADS = {
    'contributions': {
        'cycle': {'transactionTypeCode': 'TCOC', 'filingYear': '', 'type': 'csv', 'fileName': 'contributions.csv'},
        'year': {'transactionTypeCode': 'TCON', 'type': 'csv', 'fileName': 'contributions.csv'},
    },
    'expenditures': {
        'cycle': {'transactionTypeCode': 'TEXC', 'filingYear': '', 'type': 'csv', 'fileName': 'expenditures.csv'},
        'year': {'transactionTypeCode': 'TEXP', 'type': 'csv', 'fileName': 'expenditures.csv'},
    },
    'committees': {
        'cycle': {'transactionTypeCode': 'TCMC', 'filingYear': '', 'type': 'csv', 'fileName': 'committees.csv'},
        'year': {'transactionTypeCode': 'TCOM', 'type': 'csv', 'fileName': 'committees.csv'},
    },
}


def download_file(file_type, year=None, progress_callback=None):
    """Download a single CSV from CRIS.

    Args:
        file_type: 'contributions', 'expenditures', or 'committees'
        year: specific year string like '2025', or None for full cycle
        progress_callback: optional function(message) for status updates
    Returns:
        path to downloaded file, or None on failure
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    mode = 'year' if year else 'cycle'
    params = DOWNLOADS[file_type][mode].copy()
    if year:
        params['filingYear'] = str(year)

    out_path = os.path.join(RAW_DIR, params['fileName'])
    label = f"{file_type} ({'cycle' if not year else year})"

    log(f"Downloading {label}...")
    log(f"  POST {CRIS_API_URL}")

    try:
        resp = requests.post(
            CRIS_API_URL,
            json=params,
            headers={
                'Content-Type': 'application/json',
                'Accept': '*/*',
                'Origin': 'https://campaignfinance.maryland.gov',
                'Referer': 'https://campaignfinance.maryland.gov/',
            },
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()

        size = 0
        with open(out_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                size += len(chunk)
                mb = size / (1024 * 1024)
                if mb > 0 and int(mb) != int((size - len(chunk)) / (1024 * 1024)):
                    log(f"  Downloaded {mb:.1f} MB...")

        final_mb = os.path.getsize(out_path) / (1024 * 1024)
        log(f"  Saved {out_path} ({final_mb:.1f} MB)")
        return out_path

    except requests.RequestException as e:
        log(f"  ERROR downloading {label}: {e}")
        return None


def download_all(year=None, types=None, progress_callback=None):
    """Download all file types.

    Args:
        year: specific year or None for cycle
        types: list of types to download, or None for all
        progress_callback: optional status callback
    Returns:
        dict of {type: path} for successful downloads
    """
    if types is None:
        types = ['committees', 'contributions', 'expenditures']

    results = {}
    for t in types:
        path = download_file(t, year=year, progress_callback=progress_callback)
        if path:
            results[t] = path

    return results


def main():
    parser = argparse.ArgumentParser(description='Download MD campaign finance data from CRIS')
    parser.add_argument('--year', type=str, default=None,
                       help='Specific year (e.g., 2025). Default: full current cycle')
    parser.add_argument('--type', type=str, default=None,
                       choices=['contributions', 'expenditures', 'committees', 'contrib', 'expend', 'comm'],
                       help='Download only this type. Default: all')
    args = parser.parse_args()

    types = None
    if args.type:
        type_map = {
            'contrib': 'contributions', 'contributions': 'contributions',
            'expend': 'expenditures', 'expenditures': 'expenditures',
            'comm': 'committees', 'committees': 'committees',
        }
        types = [type_map[args.type]]

    results = download_all(year=args.year, types=types)

    if results:
        print(f"\nDownloaded {len(results)} file(s) to {RAW_DIR}")
        for t, p in results.items():
            print(f"  {t}: {p}")
    else:
        print("No files downloaded.")


if __name__ == '__main__':
    main()
