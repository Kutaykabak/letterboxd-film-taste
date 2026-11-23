"""Pulls a catalog of well-known films from TMDB so the site can show a predicted rating for them.

Usage (from the project root):
    py src\\fetch_catalog.py [number_of_pages]

Each page of TMDB's /discover endpoint has 20 films, sorted by vote count. The default
250 pages gives the 5,000 most-voted films. Runtime isn't in the discover results; the
model's imputer fills it with the training median.

Output: data/processed/tmdb_catalog.csv
"""
import csv
import sys
import time

from tmdb_enrich import OUT, get, load_key, make_session

PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 250


def main():
    session = make_session(load_key())
    genre_names = {g["id"]: g["name"] for g in get(session, "/genre/movie/list", language="en-US")["genres"]}

    rows, seen = [], set()
    for page in range(1, PAGES + 1):
        data = get(session, "/discover/movie", sort_by="vote_count.desc", include_adult="false",
                   include_video="false", language="en-US", page=page)
        for m in data["results"]:
            year = (m.get("release_date") or "")[:4]
            if m["id"] in seen or not year.isdigit():
                continue
            seen.add(m["id"])
            rows.append({
                "tmdb_id": m["id"],
                "name": m.get("title") or m.get("original_title"),
                "year": int(year),
                "genres": "|".join(genre_names.get(g, "") for g in m.get("genre_ids", []) if g in genre_names),
                "original_language": m.get("original_language"),
                "tmdb_vote_average": m.get("vote_average"),
                "tmdb_vote_count": m.get("vote_count"),
                "popularity": m.get("popularity"),
            })
        if page % 25 == 0:
            print(f"  {page}/{PAGES} pages, {len(rows)} films")
        time.sleep(0.03)

    out = OUT / "tmdb_catalog.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} films to {out}")


if __name__ == "__main__":
    main()
