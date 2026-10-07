"""Profile the raw Edmunds review files using only Python's standard library."""

import csv
from collections import Counter
from pathlib import Path


RAW_DIR = Path(__file__).parent / "data" / "raw"


def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    json_files = sorted((RAW_DIR / "json").glob("*.json"))
    author_counts = Counter()
    vehicle_titles = set()
    columns = None
    row_count = 0
    missing_ratings = 0
    min_rating = float("inf")
    max_rating = float("-inf")

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if columns is None:
                columns = reader.fieldnames
            for row in reader:
                row_count += 1
                author = (row.get("Author_Name") or "").strip()
                vehicle = (row.get("Vehicle_Title") or "").strip()
                rating_text = (row.get("Rating") or "").strip()

                if author:
                    author_counts[author] += 1
                if vehicle:
                    vehicle_titles.add(vehicle)
                try:
                    rating = float(rating_text)
                    min_rating = min(min_rating, rating)
                    max_rating = max(max_rating, rating)
                except ValueError:
                    missing_ratings += 1

    repeat_authors = {author: count for author, count in author_counts.items() if count >= 2}

    print(f"CSV files: {len(csv_files)}")
    print(f"JSON files: {len(json_files)}")
    print(f"Rows: {row_count:,}")
    print(f"Columns: {columns}")
    print(f"Unique authors: {len(author_counts):,}")
    print(f"Authors with at least 2 reviews: {len(repeat_authors):,}")
    print(f"Rows from repeat authors: {sum(repeat_authors.values()):,}")
    print(f"Unique vehicle titles: {len(vehicle_titles):,}")
    print(f"Missing or invalid ratings: {missing_ratings:,}")
    print(f"Rating range: {min_rating:g} to {max_rating:g}")


if __name__ == "__main__":
    main()
