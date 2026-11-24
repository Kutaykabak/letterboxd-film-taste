""""What should we watch together?" A simple joint recommender for Elif and me.

1. Matches Elif's Letterboxd films (ratings, watched, watchlist) with TMDB. It shares the
   cache with tmdb_enrich.py, so films I already have are not fetched again.
2. Fits the same Random Forest as my model on Elif's ratings and measures both models
   with repeated 5-fold cross-validation against a baseline that predicts the mean.
3. Candidates: the TMDB catalog (src/fetch_catalog.py) plus both watchlists, minus every
   film either of us has watched. Each film gets a predicted rating for each of us and is
   ranked by the worse of its two ranks (Elif rates higher overall, so raw predictions
   are not compared directly).

Outputs: data/processed/elif_enriched.csv (not in the repo, from Elif's raw export)
         reports/together_recommendations.csv
Run from the project root:  py src/build_together.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import build  # noqa: E402
import tmdb_enrich as te  # noqa: E402

PROC = ROOT / "data" / "processed"
ELIF = ROOT / "data" / "raw" / "elif"


# ---------------------------------------------------------------- 1. TMDB for Elif's films
def enrich_elif():
    films = {}
    for name in ("ratings", "watched", "watchlist"):
        path = ELIF / f"{name}.csv"
        if path.exists():
            with open(path, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    films.setdefault(row["Letterboxd URI"], (row["Name"], row["Year"]))
    cache = json.loads(te.CACHE.read_text(encoding="utf-8"))
    todo = [u for u in films if u not in cache]
    print(f"Elif: {len(films)} films, {len(todo)} not in the TMDB cache yet")
    if todo:
        session = te.make_session(te.load_key())
        for i, uri in enumerate(todo, 1):
            name, year = films[uri]
            try:
                tid = te.find_movie(session, name, year) if year else None
                if tid is not None:
                    cache[uri] = te.get(session, f"/movie/{tid}", append_to_response="credits,keywords")
            except Exception as e:  # keep going on network errors
                print("  error:", name, year, e)
            if i % 50 == 0:
                te.CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"  {i}/{len(todo)}")
            time.sleep(0.05)
        te.CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    rows = [te.flatten(u, *films[u], cache[u]) for u in films if u in cache]
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "elif_enriched.csv", index=False, encoding="utf-8-sig")
    print(f"Elif: {len(out)} of {len(films)} matched")
    return out


el_films = enrich_elif()
el_films["year"] = pd.to_numeric(el_films["year"], errors="coerce")

# ---------------------------------------------------------------- 2. one model each
me = pd.read_csv(PROC / "analysis_table.csv")
genre_cols = [c for c in me.columns if c.startswith("g_")]

el_r = pd.read_csv(ELIF / "ratings.csv")
el = el_films.merge(el_r[["Letterboxd URI", "Rating"]], left_on="uri", right_on="Letterboxd URI")
el = el.rename(columns={"Rating": "rating"}).drop_duplicates("tmdb_id")


def rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                               random_state=42, n_jobs=1))


cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
fits = {}
print("\nRepeated 5-fold CV (5x10), RMSE in stars:")
for who, d in [("me", me), ("Elif", el)]:
    X, y = build(d, genre_cols), d["rating"].values
    base = -cross_val_score(make_pipeline(SimpleImputer(), DummyRegressor()), X, y, cv=cv,
                            scoring="neg_root_mean_squared_error")
    mod = -cross_val_score(rf(), X, y, cv=cv, scoring="neg_root_mean_squared_error")
    print(f"  {who:5s} n={len(y)}  mean rating {y.mean():.2f}  baseline {base.mean():.3f}  "
          f"RF {mod.mean():.3f} (sd {mod.std():.3f})  improvement {1 - mod.mean() / base.mean():.0%}")
    fits[who] = rf().fit(X, y)

imp = pd.Series(fits["Elif"][-1].feature_importances_, index=build(el, genre_cols).columns)
print("\nElif's model, top impurity importances:", imp.sort_values(ascending=False).head(5).round(2).to_dict())

# ---------------------------------------------------------------- 3. joint ranking
cat = pd.read_csv(PROC / "tmdb_catalog.csv")
my_enriched = pd.read_csv(PROC / "films_enriched.csv")
my_wl = pd.read_csv(ROOT / "data/raw/watchlist.csv")["Letterboxd URI"]
el_wl = pd.read_csv(ELIF / "watchlist.csv")["Letterboxd URI"]
my_wl_ids = set(my_enriched.loc[my_enriched.uri.isin(my_wl), "tmdb_id"])
el_wl_ids = set(el_films.loc[el_films.uri.isin(el_wl), "tmdb_id"])

cols = ["tmdb_id", "name", "year", "genres", "original_language", "runtime",
        "tmdb_vote_average", "tmdb_vote_count", "popularity"]
wl_films = pd.concat([my_enriched[my_enriched.tmdb_id.isin(my_wl_ids)], el_films[el_films.tmdb_id.isin(el_wl_ids)]])
cand = pd.concat([cat.assign(runtime=np.nan)[cols], wl_films[cols]], ignore_index=True).drop_duplicates("tmdb_id")

# everything either of us has watched or rated
my_seen = set(my_enriched.loc[my_enriched.uri.isin(pd.concat([
    pd.read_csv(ROOT / "data/raw/watched.csv")["Letterboxd URI"],
    pd.read_csv(ROOT / "data/raw/ratings.csv")["Letterboxd URI"]])), "tmdb_id"])
el_seen = set(el_films.loc[el_films.uri.isin(pd.concat([
    pd.read_csv(ELIF / "watched.csv")["Letterboxd URI"], el_r["Letterboxd URI"]])), "tmdb_id"])
cand = cand[~cand.tmdb_id.isin(my_seen | el_seen)]
cand = cand[(cand.tmdb_vote_count >= 300) & cand.year.notna()].copy()

Xc = build(cand, genre_cols)
cand["pred_me"] = np.clip(fits["me"].predict(Xc), 0.5, 5)
cand["pred_elif"] = np.clip(fits["Elif"].predict(Xc), 0.5, 5)
# Elif rates about half a star higher than me, so the raw predictions are not on the same
# scale. Each film gets its rank among each of our predictions (top share, 0-100, lower is
# better) and is ranked by the worse of the two.
cand["top_me"] = 100 * (1 - cand["pred_me"].rank(pct=True))
cand["top_elif"] = 100 * (1 - cand["pred_elif"].rank(pct=True))
cand["together"] = cand[["top_me", "top_elif"]].max(axis=1)
cand["my_watchlist"] = cand.tmdb_id.isin(my_wl_ids)
cand["elif_watchlist"] = cand.tmdb_id.isin(el_wl_ids)
cand = cand.sort_values(["together", "tmdb_vote_average"], ascending=[True, False])

out = cand[["name", "year", "genres", "tmdb_vote_average", "pred_me", "pred_elif", "top_me", "top_elif", "together",
            "my_watchlist", "elif_watchlist"]].round(2)
out.to_csv(ROOT / "reports/together_recommendations.csv", index=False, encoding="utf-8-sig")
print(f"\n{len(cand)} candidates (watched by neither, 300+ TMDB votes)")
with pd.option_context("display.width", 200, "display.max_colwidth", 30):
    print(out.head(25).to_string(index=False))
