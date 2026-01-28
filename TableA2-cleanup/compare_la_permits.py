#!/usr/bin/env python3
"""Compare Los Angeles permit counts: calculated from APR vs HCD dashboard values.

Uses HARDFILTER method: row recovery + triplet/DEMO/date-year validation.
"""

from pathlib import Path

apr_path = Path(__file__).parent / "tablea2.csv"

# HCD Dashboard values for Los Angeles (2021-2024)
HCD_DASHBOARD_LA = {2021: 19629, 2022: 22621, 2023: 18622, 2024: 17195}

# Column indices
YEAR_COL = 2
ENT_DATE_COL = 17
ISS_DATE_COL = 26
CO_DATE_COL = 35
PERMITS_COL = 27
DEMO_COL = 44
permit_years = [2021, 2022, 2023, 2024]

def is_juris(val):
    """Return True if val is a non-empty jurisdiction code (required field)."""
    v = str(val).strip()
    return bool(v) and ',' not in v and v not in ("nan", "None")

def is_year(val):
    """Return True if val is a valid YEAR (2018-2024 only)."""
    v = str(val).strip()
    return v.isdigit() and 2018 <= int(v) <= 2024

def is_date(val):
    """Return True if val looks like a date. Primary: YYYY-MM-DD, fallback: MM/DD/YYYY."""
    v = str(val).strip()
    if not v:
        return True
    if '-' in v and len(v) == 10 and v[:4].isdigit():
        return True
    return '/' in v and 8 <= len(v) <= 10

def is_non_numeric_demo(val):
    """Return True if DEMO value is non-empty but not numeric."""
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return False
    try:
        float(v)
        return False
    except ValueError:
        return True

def is_excessive_demo(val):
    """Return True if val is numeric and > 99 (should be filtered)."""
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return False
    try:
        return int(float(v)) > 99
    except (ValueError, TypeError):
        return False

def extract_year_from_date(val):
    """Extract year from date string. Primary: YYYY-MM-DD, fallback: MM/DD/YYYY."""
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return None
    if '-' in v and len(v) >= 10 and v[:4].isdigit():
        return v[:4]
    if '/' in v:
        parts = v.split('/')
        if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
            return parts[2]
    return None

def validate_date_year(row, year_str, iss_pos, ent_pos, co_pos):
    """Validate that at least one date exists and its year matches YEAR."""
    n = len(row)
    for pos, name in [(iss_pos, "ISS_DATE"), (ent_pos, "ENT_DATE"), (co_pos, "CO_DATE")]:
        if pos < n:
            year = extract_year_from_date(row[pos])
            if year:
                return (True, None) if year == year_str else (False, f"{name} mismatch")
    return False, "All dates empty"

def find_date_position(parts, start, end):
    """Find a date anchor in parts[start:end]."""
    for i in range(start, min(end, len(parts))):
        if is_date(parts[i]) and parts[i].strip():
            return i
    return None


# Step 1: Read and join multi-line quoted fields
print(f"Loading: {apr_path}")
with open(apr_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

joined_lines = []
current_line = []
in_quote = False
for char in content:
    if char == '"':
        in_quote = not in_quote
        current_line.append(char)
    elif char == '\n':
        if in_quote:
            current_line.append(' ')
        else:
            joined_lines.append(''.join(current_line))
            current_line = []
    else:
        current_line.append(char)
if current_line:
    joined_lines.append(''.join(current_line))

if in_quote:
    print(f"WARNING: File ended with unclosed quote")

header_parts = joined_lines[0].split(',')
expected_cols = len(header_parts)
print(f"Lines: {len(joined_lines):,}, Expected columns: {expected_cols}")

# Step 2: Parse and aggregate Los Angeles permits (HARDFILTER method)
la_permits = {y: 0 for y in permit_years}
la_rows = 0
normal_count = 0
recovered_count = 0
skipped_triplet = 0
skipped_non_numeric_demo = 0
skipped_excessive_demo = 0
skipped_date_mismatch = 0
skipped_fewer_cols = 0

for line in joined_lines[1:]:
    if not line.strip():
        continue
    parts = line.split(',')
    n = len(parts)
    
    if n < 3:
        skipped_fewer_cols += 1
        continue
    
    juris = parts[0].strip().upper()
    if juris != "LOS ANGELES":
        continue
    
    # HARDFILTER: Triplet validation
    if not is_juris(parts[0]) or not is_juris(parts[1]) or not is_year(parts[2]):
        skipped_triplet += 1
        continue
    
    year_str = parts[YEAR_COL].strip()
    permits_pos = PERMITS_COL
    demo_pos = DEMO_COL
    iss_pos, ent_pos, co_pos = ISS_DATE_COL, ENT_DATE_COL, CO_DATE_COL
    
    if n > expected_cols:
        # Recovery: find ISS_DATE position
        extra = n - expected_cols
        found_iss = find_date_position(parts, ISS_DATE_COL, ISS_DATE_COL + extra + 1)
        if found_iss:
            shift = found_iss - ISS_DATE_COL
            permits_pos = PERMITS_COL + shift
            demo_pos = DEMO_COL + shift
            iss_pos = found_iss
            ent_pos = ENT_DATE_COL + shift
            co_pos = CO_DATE_COL + shift
        else:
            permits_pos = PERMITS_COL + extra
            demo_pos = DEMO_COL + extra
            iss_pos = ISS_DATE_COL + extra
            ent_pos = ENT_DATE_COL + extra
            co_pos = CO_DATE_COL + extra
        recovered_count += 1
    else:
        normal_count += 1
    
    # HARDFILTER: DEMO validation
    demo_val = parts[demo_pos] if demo_pos < n else ""
    if is_non_numeric_demo(demo_val):
        skipped_non_numeric_demo += 1
        continue
    if is_excessive_demo(demo_val):
        skipped_excessive_demo += 1
        continue
    
    # Date-year validation
    valid, reason = validate_date_year(parts, year_str, iss_pos, ent_pos, co_pos)
    if not valid:
        skipped_date_mismatch += 1
        continue
    
    try:
        year = int(float(year_str))
    except (ValueError, TypeError):
        continue
    
    if year not in permit_years:
        continue
    
    permits_str = parts[permits_pos] if permits_pos < n else "0"
    try:
        permits = int(float(permits_str)) if permits_str.strip() else 0
    except (ValueError, TypeError):
        permits = 0
    
    la_permits[year] += permits
    la_rows += 1

# Display comparison
print(f"\n{'='*70}")
print(f"HARDFILTER STATISTICS (Los Angeles only)")
print(f"{'='*70}")
print(f"LA rows kept: {la_rows:,} (normal={normal_count:,}, recovered={recovered_count:,})")
print(f"Skipped:")
print(f"  - Triplet failed:      {skipped_triplet:,}")
print(f"  - Non-numeric DEMO:    {skipped_non_numeric_demo:,}")
print(f"  - DEMO > 99:          {skipped_excessive_demo:,}")
print(f"  - Date mismatch:       {skipped_date_mismatch:,}")
print(f"  - Fewer cols:          {skipped_fewer_cols:,}")
print(f"{'='*70}")

print(f"\nLOS ANGELES PERMIT COMPARISON: Calculated vs HCD Dashboard")
print(f"{'='*70}")
print(f"{'Year':<8} {'Calculated':>15} {'HCD Dashboard':>15} {'Difference':>15}")
print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*15}")

total_calc = 0
total_hcd = sum(HCD_DASHBOARD_LA.values())

for year in permit_years:
    calc = la_permits[year]
    hcd = HCD_DASHBOARD_LA[year]
    diff = calc - hcd
    total_calc += calc
    print(f"{year:<8} {calc:>15,} {hcd:>15,} {diff:>+15,}")

print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*15}")
print(f"{'TOTAL':<8} {total_calc:>15,} {total_hcd:>15,} {total_calc - total_hcd:>+15,}")
print(f"{'='*70}")

pct_diff = (total_calc - total_hcd) / total_hcd * 100 if total_hcd > 0 else 0
print(f"\nCalculated is {pct_diff:+.1f}% {'higher' if pct_diff > 0 else 'lower'} than HCD Dashboard")
