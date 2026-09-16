"""
clean_bergen_data.py

Combine weather, Bergen City Bike, road traffic and air-quality data into one
DAILY dataset for the Information Visualization project on how rainfall
relates to mobility and air quality in Bergen.

Outputs (written next to this script):
    bergen_daily_clean.csv
    bergen_cleaning_report.txt

The original CSV files are only READ, never modified.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Files are located by pattern (searched recursively below BASE_DIR) so the
# script works whether they sit in the root folder or in subfolders, and
# tolerates small name variations like "Rain-2021-2026(1).csv".
WEATHER_PATTERN = "Rain-2021-2026*.csv"
BIKE_PATTERN = "bergen_bikes_*.csv"
TRAFFIC_PATTERN = "traffic-brgn.csv"
AIR_PATTERN = "airdp.csv"

OUTPUT_CSV = BASE_DIR / "bergen_daily_clean.csv"
REPORT_TXT = BASE_DIR / "bergen_cleaning_report.txt"

START_DATE = pd.Timestamp("2021-01-01")
LOCAL_TZ = "Europe/Oslo"

TRAFFIC_STATION = "Danmarks plass ved ladestasjon"
TRAFFIC_FELT = "Totalt"          # the all-lanes, both-directions total
MIN_COVERAGE_PCT = 90.0          # QC threshold for traffic and NO2

STAT_COLUMNS = ["rain_mm", "temp_c", "bike_trips", "traffic_count", "no2_ug_m3"]
ANALYTICAL_COLUMNS = [
    "date",
    "rain_mm",
    "temp_c",
    "bike_trips",
    "bike_avg_duration_min",
    "traffic_count",
    "no2_ug_m3",
]

warnings_log = []


def warn(message):
    """Print a warning and keep it for the report (never silently correct)."""
    print(f"WARNING: {message}")
    warnings_log.append(message)


def find_one(pattern):
    """Find exactly one source file matching a pattern below BASE_DIR."""
    matches = sorted(p for p in BASE_DIR.rglob(pattern) if p.is_file())
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' found below {BASE_DIR}")
    if len(matches) > 1:
        warn(f"Several files match '{pattern}', using {matches[0].relative_to(BASE_DIR)}")
    return matches[0]


def to_float(series):
    """Convert Norwegian-formatted numbers ('1 234,5') to floats.

    Everything is read as text first so we control the conversion. Values
    that cannot be parsed (e.g. empty strings or '-') become NaN.
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(" ", "", regex=False)  # non-breaking space thousands sep.
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ---------------------------------------------------------------------------
# 1. Weather
# ---------------------------------------------------------------------------

def load_weather():
    path = find_one(WEATHER_PATTERN)
    print(f"\n[Weather] Reading {path.relative_to(BASE_DIR)}")

    # utf-8-sig strips the byte-order mark at the start of the file.
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Tid(norsk normaltid)": "date",
        "Middeltemperatur (døgn)": "temp_c",
        "Nedbør (døgn)": "rain_mm",
    })

    # The footer row ("Data er gyldig per ...") has no parseable date, so
    # requiring a valid DD.MM.YYYY date removes it along with any other junk.
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["date"])
    print(f"  Removed {n_before - len(df)} row(s) without a valid date (metadata/footer)")

    df["temp_c"] = to_float(df["temp_c"])
    df["rain_mm"] = to_float(df["rain_mm"])

    # Keep one row per date. If duplicates disagree we warn rather than guess.
    dupes = df[df.duplicated("date", keep=False)]
    if not dupes.empty:
        conflicting = dupes.groupby("date")[["temp_c", "rain_mm"]].nunique().gt(1).any(axis=1).sum()
        warn(f"Weather has {dupes['date'].nunique()} duplicated date(s), "
             f"{conflicting} with conflicting values; keeping the first row")
    df = df.drop_duplicates("date", keep="first")

    print(f"  {len(df)} daily rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df[["date", "rain_mm", "temp_c"]]


# ---------------------------------------------------------------------------
# 2. Bergen City Bike
# ---------------------------------------------------------------------------

def load_bikes():
    files = sorted(p for p in BASE_DIR.rglob(BIKE_PATTERN) if p.is_file())
    print(f"\n[Bikes] Found {len(files)} bike CSV file(s)")
    if not files:
        raise FileNotFoundError(f"No files matching '{BIKE_PATTERN}' found below {BASE_DIR}")

    # Work out which months are present, and which are missing between the
    # earliest and latest file, so forgotten downloads are easy to spot.
    months_present = set()
    valid_files = []
    for f in files:
        try:
            year, month = f.stem.split("_")[-2:]
            months_present.add(pd.Period(f"{int(year)}-{int(month):02d}", freq="M"))
            valid_files.append(f)
        except ValueError:
            warn(f"Could not read year/month from bike filename {f.name}; file skipped")
    files = valid_files

    all_months = pd.period_range(min(months_present), max(months_present), freq="M")
    missing_months = [str(m) for m in all_months if m not in months_present]
    if missing_months:
        warn(f"Missing bike months: {', '.join(missing_months)}")
    else:
        print(f"  No missing months between {min(months_present)} and {max(months_present)}")

    frames = []
    for f in files:
        # Read as text so exact-duplicate detection compares raw values.
        part = pd.read_csv(f, dtype=str, encoding="utf-8")
        if part.empty:
            warn(f"Bike file {f.name} contains no trips")
        frames.append(part)
    trips = pd.concat(frames, ignore_index=True)
    print(f"  {len(trips):,} raw trip rows loaded")

    # Exact duplicates (identical in every column) are treated as double
    # records. Concatenating first also catches duplicates across files.
    n_before = len(trips)
    trips = trips.drop_duplicates()
    duplicates_removed = n_before - len(trips)
    print(f"  Removed {duplicates_removed:,} exact duplicate trip row(s)")

    # Timestamps are UTC. Convert to Bergen local time BEFORE taking the date,
    # otherwise trips between 00:00 and 01:00/02:00 local time would be counted
    # on the previous day.
    started_utc = pd.to_datetime(trips["started_at"], utc=True, errors="coerce", format="ISO8601")
    invalid_start = started_utc.isna().sum()
    if invalid_start:
        warn(f"{invalid_start:,} bike trip(s) without a valid started_at were ignored")
    trips = trips.assign(started_local=started_utc.dt.tz_convert(LOCAL_TZ))
    trips = trips.dropna(subset=["started_local"])
    trips["date"] = trips["started_local"].dt.tz_localize(None).dt.normalize()

    # Duration is in seconds. A missing/invalid duration only affects the
    # average duration, never the trip count.
    trips["duration_min"] = pd.to_numeric(trips["duration"], errors="coerce") / 60
    n_missing_duration = trips["duration_min"].isna().sum()
    if n_missing_duration:
        warn(f"{n_missing_duration:,} bike trip(s) have no valid duration "
             "(still counted as trips, excluded from average duration)")
    n_nonpositive = (trips["duration_min"] <= 0).sum()
    if n_nonpositive:
        warn(f"{n_nonpositive:,} bike trip(s) have duration <= 0 seconds (kept as-is)")

    daily = trips.groupby("date").agg(
        bike_trips=("date", "size"),
        bike_avg_duration_min=("duration_min", "mean"),  # mean ignores NaN
    ).reset_index()

    info = {
        "files_loaded": len(files),
        "months_present": months_present,
        "missing_months": missing_months,
        "duplicates_removed": duplicates_removed,
        "invalid_start": int(invalid_start),
        "last_date": daily["date"].max(),
    }
    print(f"  {len(daily)} days with trips, {daily['date'].min().date()} to {daily['date'].max().date()}")
    return daily, info


# ---------------------------------------------------------------------------
# 3. Road traffic
# ---------------------------------------------------------------------------

def load_traffic():
    path = find_one(TRAFFIC_PATTERN)
    print(f"\n[Traffic] Reading {path.relative_to(BASE_DIR)}")

    df = pd.read_csv(path, sep=";", dtype=str, encoding="latin1")
    df.columns = df.columns.str.strip()

    # Only one station, and only the "Totalt" row. The file also contains
    # per-lane rows (1..5) and per-direction totals; adding those would count
    # the same vehicles two or three times.
    df = df[(df["Navn"].str.strip() == TRAFFIC_STATION) & (df["Felt"].str.strip() == TRAFFIC_FELT)].copy()
    if df.empty:
        raise ValueError(f"No traffic rows for Navn == '{TRAFFIC_STATION}' and Felt == '{TRAFFIC_FELT}'")

    df = df.rename(columns={
        "Dato": "date",
        "Trafikkmengde": "traffic_count",
        "Dekningsgrad (%)": "traffic_coverage_pct",
    })
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["traffic_count"] = to_float(df["traffic_count"])
    df["traffic_coverage_pct"] = to_float(df["traffic_coverage_pct"])

    if df["date"].duplicated().any():
        warn(f"Traffic has {df['date'].duplicated().sum()} duplicated date(s) for the Totalt row; keeping the first")
        df = df.drop_duplicates("date", keep="first")

    # QC: a day with less than 90 % coverage is an incomplete count and would
    # look like a traffic drop, so the value is removed (not interpolated).
    # A count with unknown coverage cannot be verified and is removed too.
    low = df["traffic_count"].notna() & (df["traffic_coverage_pct"] < MIN_COVERAGE_PCT)
    unknown = df["traffic_count"].notna() & df["traffic_coverage_pct"].isna()
    df["traffic_rejected"] = low | unknown
    df.loc[df["traffic_rejected"], "traffic_count"] = float("nan")
    print(f"  {len(df)} daily Totalt rows, {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Rejected {low.sum()} count(s) with coverage < {MIN_COVERAGE_PCT:g}%")
    if unknown.any():
        warn(f"Rejected {unknown.sum()} traffic count(s) with missing coverage")

    return df[["date", "traffic_count", "traffic_coverage_pct", "traffic_rejected"]]


# ---------------------------------------------------------------------------
# 4. Air quality
# ---------------------------------------------------------------------------

def load_air():
    path = find_one(AIR_PATTERN)
    print(f"\n[Air quality] Reading {path.relative_to(BASE_DIR)}")

    # The first three lines are NILU metadata (QC/QA explanation, source).
    df = pd.read_csv(path, sep=";", dtype=str, skiprows=3, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Find the NO2 column by content instead of the exact label, because the
    # "µg/m³" characters can vary slightly between exports.
    no2_cols = [c for c in df.columns if "NO2" in c]
    if len(no2_cols) != 1:
        raise ValueError(f"Expected exactly one NO2 column, found: {no2_cols}")

    df = df.rename(columns={"Tid": "date", no2_cols[0]: "no2_ug_m3", "Dekning": "no2_coverage_pct"})
    # Daily values are stamped at 00:00, so the day is the date part.
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%d.%m.%Y %H:%M", errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["no2_ug_m3"] = to_float(df["no2_ug_m3"])
    df["no2_coverage_pct"] = to_float(df["no2_coverage_pct"])

    if df["date"].duplicated().any():
        warn(f"Air quality has {df['date'].duplicated().sum()} duplicated date(s); keeping the first")
        df = df.drop_duplicates("date", keep="first")

    # QC: daily means based on < 90 % of the hours are not representative.
    low = df["no2_ug_m3"].notna() & (df["no2_coverage_pct"] < MIN_COVERAGE_PCT)
    unknown = df["no2_ug_m3"].notna() & df["no2_coverage_pct"].isna()
    df["no2_rejected"] = low | unknown
    df.loc[df["no2_rejected"], "no2_ug_m3"] = float("nan")
    print(f"  {len(df)} daily rows, {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Rejected {low.sum()} NO2 value(s) with coverage < {MIN_COVERAGE_PCT:g}%")
    if unknown.any():
        warn(f"Rejected {unknown.sum()} NO2 value(s) with missing coverage")

    return df[["date", "no2_ug_m3", "no2_coverage_pct", "no2_rejected"]]


# ---------------------------------------------------------------------------
# 5. Merge
# ---------------------------------------------------------------------------

def choose_end_date(weather, bike_info, traffic, air):
    """Latest date covered by all major datasets, capped at the last valid NO2."""
    last_dates = {
        "weather (last valid rain)": weather.loc[weather["rain_mm"].notna(), "date"].max(),
        "bikes (last trip)": bike_info["last_date"],
        "traffic (last valid count)": traffic.loc[traffic["traffic_count"].notna(), "date"].max(),
        "NO2 (last valid value)": air.loc[air["no2_ug_m3"].notna(), "date"].max(),
    }
    print("\n[Merge] Last available date per dataset:")
    for name, d in last_dates.items():
        print(f"  {name:28s} {d.date()}")
    return min(last_dates.values())


def add_derived(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month_name()
    df["month_number"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.day_name()
    df["weekday_number"] = df["date"].dt.weekday  # Monday = 0 ... Sunday = 6
    df["is_weekend"] = df["weekday_number"] >= 5

    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df["month_number"].map(season_map)

    # Rain flags stay empty when rain itself is missing (nullable boolean),
    # so a missing measurement is never mistaken for a dry day.
    df["is_rainy"] = (df["rain_mm"] > 0).astype("boolean")
    df.loc[df["rain_mm"].isna(), "is_rainy"] = pd.NA

    # Project analysis bins, not official meteorological classes.
    # Missing rain leaves the category empty.
    rain = df["rain_mm"]
    df["rain_category"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[rain == 0, "rain_category"] = "Dry"
    df.loc[(rain > 0) & (rain < 5), "rain_category"] = "Light"
    df.loc[(rain >= 5) & (rain < 20), "rain_category"] = "Moderate"
    df.loc[rain >= 20, "rain_category"] = "Heavy"
    return df


# ---------------------------------------------------------------------------
# 9. Validation
# ---------------------------------------------------------------------------

def validate(df, start, end):
    print("\n[Validation]")
    expected_days = (end - start).days + 1
    checks = {
        "exactly one row per date": len(df) == expected_days and df["date"].nunique() == len(df),
        "dates sorted ascending": df["date"].is_monotonic_increasing,
        "no duplicate dates": not df["date"].duplicated().any(),
    }
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            warn(f"Validation failed: {name}")

    for col in ["rain_mm", "bike_trips", "traffic_count", "no2_ug_m3"]:
        n_negative = (df[col] < 0).sum()
        print(f"  {'OK  ' if n_negative == 0 else 'FAIL'} {col} never negative")
        if n_negative:
            dates = ", ".join(df.loc[df[col] < 0, "date"].dt.strftime("%Y-%m-%d").head(10))
            warn(f"{col} has {n_negative} negative value(s) (not corrected), e.g. {dates}")

    # Plausibility checks: flagged only, never changed.
    if (df["rain_mm"] > 150).any():
        warn(f"rain_mm > 150 mm on {(df['rain_mm'] > 150).sum()} day(s)")
    if ((df["temp_c"] < -30) | (df["temp_c"] > 35)).any():
        warn("temp_c outside -30..35 °C for at least one day")
    if (df["bike_avg_duration_min"] > 180).any():
        warn(f"bike_avg_duration_min > 180 min on {(df['bike_avg_duration_min'] > 180).sum()} day(s)")


# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------

def write_report(df, start, end, bike_info, traffic_rejected, air_rejected):
    lines = [
        "BERGEN DAILY DATA - CLEANING REPORT",
        "=" * 40,
        "",
        f"Date range:  {start.date()} to {end.date()}",
        f"Rows:        {len(df)}",
        f"Columns:     {len(df.columns)}",
        "",
        "BIKES",
        f"  Bike files loaded:           {bike_info['files_loaded']}",
        f"  Missing bike months:         {', '.join(bike_info['missing_months']) or 'none'}",
        f"  Duplicate bike trips removed: {bike_info['duplicates_removed']}",
        f"  Trips without valid started_at ignored: {bike_info['invalid_start']}",
        "",
        "QUALITY CONTROL (coverage < 90% -> value set to NaN, within final date range)",
        f"  Traffic observations rejected: {traffic_rejected}",
        f"  NO2 observations rejected:     {air_rejected}",
        "",
        "MISSING VALUES PER COLUMN",
    ]
    for col, n in df.isna().sum().items():
        lines.append(f"  {col:24s} {n}")

    lines += ["", "SUMMARY STATISTICS"]
    stats = df[STAT_COLUMNS].agg(["min", "max", "mean", "median"]).T.round(2)
    lines += ["  " + line for line in stats.to_string().splitlines()]

    lines += ["", "WARNINGS"]
    lines += [f"  - {w}" for w in warnings_log] or ["  none"]

    lines += [
        "",
        "NOTES",
        "  - No measurements were imputed or interpolated; missing values stay NaN.",
        "  - bike_trips is 0 only for days in a month whose bike file exists but has no trips;",
        "    days in months without a file are NaN.",
        "  - Bike trips are dated by LOCAL (Europe/Oslo) start time.",
        "  - Traffic: Danmarks plass ved ladestasjon, Felt == 'Totalt' only.",
        "  - rain_category bins (Dry 0, Light <5, Moderate 5-<20, Heavy >=20 mm) are project",
        "    analysis bins, not official meteorological classifications.",
    ]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    weather = load_weather()
    bikes, bike_info = load_bikes()
    traffic = load_traffic()
    air = load_air()

    end_date = choose_end_date(weather, bike_info, traffic, air)
    print(f"\nChosen START_DATE: {START_DATE.date()}")
    print(f"Chosen END_DATE:   {end_date.date()}")

    # A complete calendar is the backbone, so days missing from every source
    # still get a row (with NaN values) instead of silently disappearing.
    df = pd.DataFrame({"date": pd.date_range(START_DATE, end_date, freq="D")})
    df = df.merge(weather, on="date", how="left", validate="one_to_one")
    df = df.merge(bikes, on="date", how="left", validate="one_to_one")
    df = df.merge(traffic, on="date", how="left", validate="one_to_one")
    df = df.merge(air, on="date", how="left", validate="one_to_one")

    # bike_trips: a day with no trip rows is a genuine 0 only if that month's
    # file was downloaded. If the file is missing we know nothing -> NaN.
    month_has_file = df["date"].dt.to_period("M").isin(bike_info["months_present"])
    df.loc[month_has_file & df["bike_trips"].isna(), "bike_trips"] = 0
    df["bike_trips"] = df["bike_trips"].astype("Int64")

    traffic_rejected = int(df["traffic_rejected"].fillna(False).astype(bool).sum())
    air_rejected = int(df["no2_rejected"].fillna(False).astype(bool).sum())

    df = df[ANALYTICAL_COLUMNS]
    df = add_derived(df.copy())
    df = df.sort_values("date").reset_index(drop=True)

    validate(df, START_DATE, end_date)

    print("\nFirst 10 rows:")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.head(10))
    print(f"\nFinal shape: {df.shape}")

    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_CSV, index=False, sep=",", decimal=".", encoding="utf-8")
    write_report(df, START_DATE, end_date, bike_info, traffic_rejected, air_rejected)

    print(f"\nSaved {OUTPUT_CSV.name} and {REPORT_TXT.name}")


if __name__ == "__main__":
    main()
