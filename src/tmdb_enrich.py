"""Letterboxd filmlerini TMDB ile eşleştirip zenginleştirir.

Kullanım (proje kökünden):
    py src\tmdb_enrich.py

Gereken: proje kökünde .env dosyası ->  TMDB_API_KEY=xxxxxxxx
(v3 API anahtarı ya da v4 "Read Access Token" ikisi de olur)

Çıktılar:
    data/processed/tmdb_cache.json     her film için ham TMDB yanıtı (tekrar çalıştırmada API'ye gitmez)
    data/processed/films_enriched.csv  eşleşen filmler, düz tablo
    data/processed/unmatched.csv       eşleşmeyenler -> elle düzeltilecek
"""
import csv
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CACHE = OUT / "tmdb_cache.json"
API = "https://api.themoviedb.org/3"


def load_key() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("TMDB_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    sys.exit("HATA: .env dosyasında TMDB_API_KEY bulunamadı.")


def make_session(key: str) -> requests.Session:
    s = requests.Session()
    if len(key) > 40:  # v4 token
        s.headers["Authorization"] = f"Bearer {key}"
    else:
        s.params = {"api_key": key}
    return s


def get(session, path, **params):
    for attempt in range(4):
        r = session.get(f"{API}{path}", params=params, timeout=20)
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("TMDB hız sınırı aşıldı")


def read_films():
    films = {}
    for name in ("ratings", "watched", "watchlist", "likes_films"):
        with open(RAW / f"{name}.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                films.setdefault(row["Letterboxd URI"], (row["Name"], row["Year"]))
    return films


def find_movie(session, title, year):
    """Önce yıl filtresiyle, olmazsa ±1 yıl toleransla ara."""
    res = get(session, "/search/movie", query=title, year=year)["results"]
    if not res:
        res = get(session, "/search/movie", query=title)["results"]
        res = [m for m in res if (m.get("release_date") or "0000")[:4].isdigit()
               and abs(int(m["release_date"][:4]) - int(year)) <= 1]
    return res[0]["id"] if res else None


def flatten(uri, name, year, d):
    crew = d.get("credits", {}).get("crew", [])
    cast = d.get("credits", {}).get("cast", [])
    return {
        "uri": uri, "name": name, "year": year, "tmdb_id": d["id"],
        "original_title": d.get("original_title"),
        "original_language": d.get("original_language"),
        "runtime": d.get("runtime"),
        "genres": "|".join(g["name"] for g in d.get("genres", [])),
        "countries": "|".join(c["iso_3166_1"] for c in d.get("production_countries", [])),
        "directors": "|".join(c["name"] for c in crew if c.get("job") == "Director"),
        "top_cast": "|".join(c["name"] for c in cast[:5]),
        "keywords": "|".join(k["name"] for k in d.get("keywords", {}).get("keywords", [])),
        "tmdb_vote_average": d.get("vote_average"),
        "tmdb_vote_count": d.get("vote_count"),
        "popularity": d.get("popularity"),
        "overview": (d.get("overview") or "").replace("\n", " "),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    session = make_session(load_key())
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    films = read_films()
    print(f"{len(films)} benzersiz film")

    unmatched = []
    for i, (uri, (name, year)) in enumerate(films.items(), 1):
        if uri in cache:
            continue
        try:
            tid = find_movie(session, name, year)
            if tid is None:
                unmatched.append((uri, name, year))
                continue
            cache[uri] = get(session, f"/movie/{tid}", append_to_response="credits,keywords")
        except Exception as e:  # ağ hatasında ilerlemeyi kaybetme
            print("hata:", name, year, e)
            unmatched.append((uri, name, year))
        if i % 25 == 0:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"  {i}/{len(films)}")
        time.sleep(0.05)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    rows = [flatten(u, *films[u], cache[u]) for u in films if u in cache]
    with open(OUT / "films_enriched.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "unmatched.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uri", "name", "year"])
        w.writerows(unmatched)
    print(f"Eşleşen: {len(rows)}  Eşleşmeyen: {len(unmatched)}")


if __name__ == "__main__":
    main()
