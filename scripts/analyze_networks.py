"""
Maryland Campaign Finance Network Analysis
Analyzes donor clusters and vendor networks to map political power structures.
Focus: Montgomery County, MD (2021-2026)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import csv

# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Montgomery County cities/areas for filtering
MOCO_LOCATIONS = [
    'montgomery', 'bethesda', 'rockville', 'silver spring', 'gaithersburg',
    'germantown', 'chevy chase', 'potomac', 'takoma park', 'wheaton',
    'kensington', 'olney', 'damascus', 'poolesville', 'clarksburg',
    'burtonsville', 'colesville', 'aspen hill', 'north bethesda'
]


def load_committees():
    """Load committee data to map IDs to names and metadata."""
    print("Loading committees...")
    
    # Find committee file
    committee_files = list(DATA_DIR.glob("*Committee*Download*.csv"))
    if not committee_files:
        print("WARNING: No committee file found")
        return pd.DataFrame()
    
    # Skip the header row that contains download timestamp
    df = pd.read_csv(committee_files[0], skiprows=1, low_memory=False)
    print(f"  Loaded {len(df)} committees")
    return df


def load_contributions():
    """Load and combine all contribution files."""
    print("Loading contributions...")
    
    contrib_files = list(DATA_DIR.glob("*Contribution*Download*.csv"))
    if not contrib_files:
        print("WARNING: No contribution files found")
        return pd.DataFrame()
    
    dfs = []
    for f in contrib_files:
        print(f"  Loading {f.name}...")
        try:
            # Skip header row with timestamp
            df = pd.read_csv(f, skiprows=1, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    print(f"  Combined: {before} rows, deduplicated to {after}")
    
    return combined


def load_expenditures():
    """Load and combine all expenditure files."""
    print("Loading expenditures...")
    
    expend_files = list(DATA_DIR.glob("*Expenditure*Download*.csv"))
    if not expend_files:
        print("WARNING: No expenditure files found")
        return pd.DataFrame()
    
    dfs = []
    for f in expend_files:
        print(f"  Loading {f.name}...")
        try:
            df = pd.read_csv(f, skiprows=1, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    print(f"  Combined: {before} rows, deduplicated to {after}")
    
    return combined


def is_moco_related(row, columns_to_check):
    """Check if a row is related to Montgomery County."""
    for col in columns_to_check:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).lower()
            if any(loc in val for loc in MOCO_LOCATIONS):
                return True
    return False


def filter_moco_committees(committees_df):
    """Filter to Montgomery County-related committees."""
    if committees_df.empty:
        return committees_df
    
    print("Filtering to Montgomery County committees...")
    
    # Check jurisdiction column first
    if 'Jurisdiction' in committees_df.columns:
        moco_mask = committees_df['Jurisdiction'].str.lower().str.contains('montgomery', na=False)
    else:
        moco_mask = pd.Series([False] * len(committees_df))
    
    # Also check address columns
    address_cols = [col for col in committees_df.columns if 'city' in col.lower() or 'address' in col.lower()]
    for col in address_cols:
        if col in committees_df.columns:
            moco_mask |= committees_df[col].astype(str).str.lower().str.contains('|'.join(MOCO_LOCATIONS), na=False)
    
    filtered = committees_df[moco_mask].copy()
    print(f"  Found {len(filtered)} MoCo-related committees")
    return filtered


def analyze_donor_network(contributions_df, moco_committee_ids):
    """
    Build donor-to-candidate network.
    Identify donors who give to multiple MoCo candidates.
    """
    print("\nAnalyzing donor network...")
    
    if contributions_df.empty:
        print("  No contribution data")
        return {}
    
    # Filter to MoCo committees
    id_col = None
    for col in contributions_df.columns:
        if 'filing' in col.lower() and 'id' in col.lower():
            id_col = col
            break
    
    if id_col is None:
        print("  Cannot find committee ID column")
        return {}
    
    moco_contribs = contributions_df[contributions_df[id_col].isin(moco_committee_ids)].copy()
    print(f"  {len(moco_contribs)} contributions to MoCo committees")
    
    if moco_contribs.empty:
        return {}
    
    # Find donor name column
    donor_col = None
    for col in moco_contribs.columns:
        if 'contributor' in col.lower() and 'name' in col.lower():
            donor_col = col
            break
    
    if donor_col is None:
        print("  Cannot find donor name column")
        return {}
    
    # Find committee name column
    committee_col = None
    for col in moco_contribs.columns:
        if 'committee' in col.lower() and 'name' in col.lower():
            committee_col = col
            break
    
    # Build donor -> committees mapping
    donor_committees = defaultdict(set)
    donor_totals = defaultdict(float)
    
    amount_col = None
    for col in moco_contribs.columns:
        if 'amount' in col.lower():
            amount_col = col
            break
    
    for _, row in moco_contribs.iterrows():
        donor = str(row.get(donor_col, '')).strip()
        committee = str(row.get(committee_col, row.get(id_col, ''))).strip()
        
        if donor and committee and donor.lower() not in ['nan', '']:
            donor_committees[donor].add(committee)
            if amount_col and pd.notna(row.get(amount_col)):
                try:
                    donor_totals[donor] += float(row[amount_col])
                except:
                    pass
    
    # Find donors who give to multiple committees
    multi_donors = {
        donor: {
            'committees': list(comms),
            'num_committees': len(comms),
            'total_donated': donor_totals.get(donor, 0)
        }
        for donor, comms in donor_committees.items()
        if len(comms) >= 2
    }
    
    # Sort by number of committees
    multi_donors = dict(sorted(multi_donors.items(), key=lambda x: x[1]['num_committees'], reverse=True))
    
    print(f"  Found {len(multi_donors)} donors giving to 2+ MoCo committees")
    
    return multi_donors


def analyze_vendor_network(expenditures_df, moco_committee_ids):
    """
    Build vendor-to-campaign network.
    Identify vendors used by multiple MoCo campaigns.
    """
    print("\nAnalyzing vendor network...")
    
    if expenditures_df.empty:
        print("  No expenditure data")
        return {}
    
    # Filter to MoCo committees
    id_col = None
    for col in expenditures_df.columns:
        if 'filing' in col.lower() and 'id' in col.lower():
            id_col = col
            break
    
    if id_col is None:
        print("  Cannot find committee ID column")
        return {}
    
    moco_expends = expenditures_df[expenditures_df[id_col].isin(moco_committee_ids)].copy()
    print(f"  {len(moco_expends)} expenditures by MoCo committees")
    
    if moco_expends.empty:
        return {}
    
    # Find payee/vendor name column
    vendor_col = None
    for col in moco_expends.columns:
        if any(term in col.lower() for term in ['payee', 'vendor', 'recipient']):
            if 'name' in col.lower() or vendor_col is None:
                vendor_col = col
    
    if vendor_col is None:
        print("  Cannot find vendor name column")
        print(f"  Available columns: {list(moco_expends.columns)}")
        return {}
    
    # Find committee name column
    committee_col = None
    for col in moco_expends.columns:
        if 'committee' in col.lower() and 'name' in col.lower():
            committee_col = col
            break
    
    # Build vendor -> committees mapping
    vendor_committees = defaultdict(set)
    vendor_totals = defaultdict(float)
    vendor_purposes = defaultdict(set)
    
    amount_col = None
    for col in moco_expends.columns:
        if 'amount' in col.lower():
            amount_col = col
            break
    
    purpose_col = None
    for col in moco_expends.columns:
        if 'purpose' in col.lower() or 'description' in col.lower():
            purpose_col = col
            break
    
    for _, row in moco_expends.iterrows():
        vendor = str(row.get(vendor_col, '')).strip()
        committee = str(row.get(committee_col, row.get(id_col, ''))).strip()
        
        if vendor and committee and vendor.lower() not in ['nan', '']:
            vendor_committees[vendor].add(committee)
            if amount_col and pd.notna(row.get(amount_col)):
                try:
                    vendor_totals[vendor] += float(row[amount_col])
                except:
                    pass
            if purpose_col and pd.notna(row.get(purpose_col)):
                vendor_purposes[vendor].add(str(row[purpose_col]))
    
    # Find vendors used by multiple committees (shared consultants, etc.)
    shared_vendors = {
        vendor: {
            'committees': list(comms),
            'num_committees': len(comms),
            'total_paid': vendor_totals.get(vendor, 0),
            'purposes': list(vendor_purposes.get(vendor, set()))[:5]  # Top 5 purposes
        }
        for vendor, comms in vendor_committees.items()
        if len(comms) >= 2
    }
    
    # Sort by number of committees
    shared_vendors = dict(sorted(shared_vendors.items(), key=lambda x: x[1]['num_committees'], reverse=True))
    
    print(f"  Found {len(shared_vendors)} vendors serving 2+ MoCo committees")
    
    return shared_vendors


def identify_power_clusters(donor_network, vendor_network):
    """
    Identify likely power clusters based on overlapping donors and vendors.
    """
    print("\nIdentifying power clusters...")
    
    # Committees that share donors
    committee_donor_overlap = defaultdict(lambda: defaultdict(int))
    for donor, info in donor_network.items():
        comms = info['committees']
        for i, c1 in enumerate(comms):
            for c2 in comms[i+1:]:
                committee_donor_overlap[c1][c2] += 1
                committee_donor_overlap[c2][c1] += 1
    
    # Committees that share vendors
    committee_vendor_overlap = defaultdict(lambda: defaultdict(int))
    for vendor, info in vendor_network.items():
        comms = info['committees']
        for i, c1 in enumerate(comms):
            for c2 in comms[i+1:]:
                committee_vendor_overlap[c1][c2] += 1
                committee_vendor_overlap[c2][c1] += 1
    
    # Combined overlap score
    all_committees = set(committee_donor_overlap.keys()) | set(committee_vendor_overlap.keys())
    
    cluster_links = []
    for c1 in all_committees:
        for c2 in all_committees:
            if c1 < c2:
                donor_overlap = committee_donor_overlap.get(c1, {}).get(c2, 0)
                vendor_overlap = committee_vendor_overlap.get(c1, {}).get(c2, 0)
                if donor_overlap > 0 or vendor_overlap > 0:
                    cluster_links.append({
                        'committee_1': c1,
                        'committee_2': c2,
                        'shared_donors': donor_overlap,
                        'shared_vendors': vendor_overlap,
                        'total_connections': donor_overlap + vendor_overlap
                    })
    
    # Sort by total connections
    cluster_links.sort(key=lambda x: x['total_connections'], reverse=True)
    
    print(f"  Found {len(cluster_links)} committee pairs with shared donors/vendors")
    
    return cluster_links


def generate_reports(committees, donor_network, vendor_network, cluster_links):
    """Generate output reports."""
    print("\nGenerating reports...")
    
    # 1. Top donors
    top_donors = []
    for donor, info in list(donor_network.items())[:100]:
        top_donors.append({
            'donor': donor,
            'num_committees': info['num_committees'],
            'total_donated': info['total_donated'],
            'committees': '; '.join(info['committees'][:10])
        })
    
    if top_donors:
        pd.DataFrame(top_donors).to_csv(OUTPUT_DIR / 'top_donors_moco.csv', index=False)
        print(f"  Saved top_donors_moco.csv ({len(top_donors)} donors)")
    
    # 2. Top vendors
    top_vendors = []
    for vendor, info in list(vendor_network.items())[:100]:
        top_vendors.append({
            'vendor': vendor,
            'num_committees': info['num_committees'],
            'total_paid': info['total_paid'],
            'committees': '; '.join(info['committees'][:10]),
            'purposes': '; '.join(info['purposes'][:3])
        })
    
    if top_vendors:
        pd.DataFrame(top_vendors).to_csv(OUTPUT_DIR / 'top_vendors_moco.csv', index=False)
        print(f"  Saved top_vendors_moco.csv ({len(top_vendors)} vendors)")
    
    # 3. Committee clusters
    if cluster_links:
        pd.DataFrame(cluster_links[:200]).to_csv(OUTPUT_DIR / 'committee_clusters_moco.csv', index=False)
        print(f"  Saved committee_clusters_moco.csv ({min(200, len(cluster_links))} pairs)")
    
    # 4. Summary stats
    summary = {
        'total_moco_committees': len(committees) if not committees.empty else 0,
        'donors_giving_to_multiple': len(donor_network),
        'vendors_serving_multiple': len(vendor_network),
        'committee_cluster_pairs': len(cluster_links),
        'top_connected_donor': list(donor_network.keys())[0] if donor_network else 'N/A',
        'top_connected_vendor': list(vendor_network.keys())[0] if vendor_network else 'N/A',
    }
    
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary.json")
    
    # 5. Full networks as JSON for visualization
    with open(OUTPUT_DIR / 'donor_network.json', 'w') as f:
        json.dump(donor_network, f, indent=2)
    
    with open(OUTPUT_DIR / 'vendor_network.json', 'w') as f:
        json.dump(vendor_network, f, indent=2)
    
    print("  Saved network JSON files")


def main():
    print("=" * 60)
    print("Maryland Campaign Finance Network Analysis")
    print("Focus: Montgomery County, 2021-2026")
    print("=" * 60)
    
    # Load data
    committees = load_committees()
    contributions = load_contributions()
    expenditures = load_expenditures()
    
    # Filter to Montgomery County
    moco_committees = filter_moco_committees(committees)
    
    if moco_committees.empty:
        print("\nNo Montgomery County committees found. Analyzing all data...")
        moco_committee_ids = set(committees['Filing Entity Id'].astype(str)) if 'Filing Entity Id' in committees.columns else set()
    else:
        moco_committee_ids = set(moco_committees['Filing Entity Id'].astype(str)) if 'Filing Entity Id' in moco_committees.columns else set()
    
    print(f"\nTracking {len(moco_committee_ids)} committee IDs")
    
    # Analyze networks
    donor_network = analyze_donor_network(contributions, moco_committee_ids)
    vendor_network = analyze_vendor_network(expenditures, moco_committee_ids)
    cluster_links = identify_power_clusters(donor_network, vendor_network)
    
    # Generate reports
    generate_reports(moco_committees, donor_network, vendor_network, cluster_links)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Results in output/ directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
