from pathlib import Path
from urllib.request import urlretrieve

start_year = 2021
start_month = 1

end_year = 2026
end_month = 1

folder = Path("bergen_bikes")
folder.mkdir(exist_ok=True)

year = start_year
month = start_month

while (year, month) <= (end_year, end_month):
    url = (
        f"https://data.urbansharing.com/"
        f"bergenbysykkel.no/trips/v1/{year}/{month:02d}.csv"
    )

    filename = folder / f"bergen_bikes_{year}_{month:02d}.csv"

    print(f"Downloading {year}-{month:02d}...")

    try:
        urlretrieve(url, filename)
        print(f"  saved: {filename}")
    except Exception as e:
        print(f"  ERROR: {e}")

    month += 1

    if month == 13:
        month = 1
        year += 1

print("Done!")
