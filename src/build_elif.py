"""Writes site/elif.js: aggregated agreement data between my ratings and Elif's for the site.

Only aggregates leave this script (rating-pair counts, genre averages, a few statistics),
not Elif's film-by-film ratings. Her raw export is read from data/raw/elif/ratings.csv,
which is not part of the repository.

Run from the project root:  py src/build_elif.py
"""
import json
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

me = pd.read_csv(ROOT / "data/processed/analysis_table.csv")
el = pd.read_csv(ROOT / "data/raw/elif/ratings.csv").rename(columns={"Name": "name", "Year": "year", "Rating": "rating_elif"})
df = me.merge(el[["name", "year", "rating_elif"]], on=["name", "year"]).rename(columns={"rating": "rating_me"})
df["diff"] = df.rating_elif - df.rating_me

pairs = df.groupby(["rating_me", "rating_elif"]).size().reset_index(name="n")

genres = []
for g in [c for c in df.columns if c.startswith("g_")]:
    sub = df[df[g] == 1]
    if len(sub) >= 15:
        genres.append({"name": g[2:], "n": int(len(sub)),
                       "me": round(sub.rating_me.mean(), 3), "el": round(sub.rating_elif.mean(), 3)})
genres.sort(key=lambda r: r["el"] - r["me"], reverse=True)

out = {
    "n": int(len(df)),
    "rho": round(stats.spearmanr(df.rating_me, df.rating_elif).statistic, 3),
    "rho_tmdb": round(stats.spearmanr(df.rating_me, df.tmdb_vote_average).statistic, 3),
    "diff": round(df["diff"].mean(), 3),
    "same": round((df["diff"] == 0).mean(), 3),
    "half": round((df["diff"].abs() <= 0.5).mean(), 3),
    "pairs": [[float(a), float(b), int(n)] for a, b, n in pairs.itertuples(index=False)],
    "genres": genres,
}
(ROOT / "site/elif.js").write_text("window.ELIF = " + json.dumps(out, ensure_ascii=False) + ";\n", encoding="utf-8")
print(f"site/elif.js: {out['n']} shared films, {len(out['pairs'])} rating pairs, {len(genres)} genres")
