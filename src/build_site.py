"""site/data.js dosyasını üretir (statik sitenin verisi).

Kullanım (proje kökünden):
    py src\\build_site.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
OUT = ROOT / 'site' / 'data.js'

rng = np.random.default_rng(42)
df = pd.read_csv(PROC / 'analysis_table.csv')
genre_cols = [c for c in df.columns if c.startswith('g_')]


def first(s):
    return str(s).split('|')[0] if isinstance(s, str) and s else ''


films = [{
    'n': r['name'], 'y': int(r['year']), 'r': float(r['rating']),
    't': round(float(r['tmdb_vote_average']), 2), 'd': first(r['directors']),
    'g': r['genres'] if isinstance(r['genres'], str) else '', 'l': int(r['liked'])
} for _, r in df.iterrows()]

# on yıllar
grp = np.where(df['decade'] < 1980, 1970, df['decade'])
decades = []
for d in sorted(set(grp)):
    x = df.loc[grp == d, 'rating']
    first_d = d == 1970
    decades.append({
        'from': 1925 if first_d else int(d), 'to': 1980 if first_d else int(d) + 10,
        'label_tr': '1980 öncesi' if first_d else f"{d}'ler",
        'label_en': 'Before 1980' if first_d else f"{d}s",
        'mean': round(x.mean(), 2), 'n': int(len(x))})


def boot(x, n=4000):
    m = rng.choice(np.asarray(x), size=(n, len(x)), replace=True).mean(axis=1)
    return [round(float(np.quantile(m, .025)), 2), round(float(np.quantile(m, .975)), 2)]


genres = []
for c in genre_cols:
    x = df.loc[df[c] == 1, 'rating']
    if len(x) >= 15:
        lo, hi = boot(x)
        genres.append({'name': c[2:], 'mean': round(x.mean(), 2), 'lo': lo, 'hi': hi, 'n': int(len(x))})
genres.sort(key=lambda g: -g['mean'])

# kalabalıktan ayrıldığım filmler: puan ~ TMDB doğrusal uyumundan artıklar
b, a = np.polyfit(df['tmdb_vote_average'], df['rating'], 1)
exp = a + b * df['tmdb_vote_average']
res = df['rating'] - exp
def pick(idx):
    return [{'n': df.loc[i, 'name'], 'y': int(df.loc[i, 'year']), 'r': float(df.loc[i, 'rating']),
             't': round(float(df.loc[i, 'tmdb_vote_average']), 1), 'e': round(float(exp[i]), 1)} for i in idx]
above = pick(res.sort_values(ascending=False).index[:6])
below = pick(res.sort_values().index[:6])

# model sonuçları: notebooks/03_modelleme.ipynb (5x10 tekrarlı CV)
models = [
    {'tr': t, 'en': e, 'rmse': r, 'sd': s, 'base': e.startswith('Baseline')} for t, e, r, s in [
        ('Temel çizgi (ortalama)', 'Baseline (mean)', 0.910, 0.070),
        ('Sadece TMDB puanı', 'TMDB rating only', 0.703, 0.057),
        ('Ridge', 'Ridge', 0.686, 0.047),
        ('Spline + Ridge', 'Spline + Ridge', 0.685, 0.053),
        ('Film yılı + TMDB puanı', 'Release year + TMDB rating', 0.682, 0.054),
        ('Lasso', 'Lasso', 0.671, 0.051),
        ('XGBoost', 'XGBoost', 0.665, 0.056),
        ('Gradient Boosting', 'Gradient Boosting', 0.664, 0.056),
        ('LightGBM', 'LightGBM', 0.662, 0.058),
        ('Sıralı lojistik', 'Ordinal logistic', 0.656, 0.056),
        ('Random Forest', 'Random Forest', 0.649, 0.054),
    ]
]

wl = pd.read_csv(ROOT / 'reports' / 'izleme_listesi_tahminleri.csv')
watch = [{'n': r['name'], 'y': int(r['year']), 'p': float(r['tahmin']),
          'g': r['genres'] if isinstance(r['genres'], str) else '', 'd': first(r['directors'])}
         for _, r in wl.iterrows()]

data = {
    'films': films, 'decades': decades, 'genres': genres,
    'fit': {'a': round(float(a), 4), 'b': round(float(b), 4)},
    'above': above, 'below': below, 'models': models, 'watch': watch,
    'stats': {'n': int(len(df)), 'mean': round(df['rating'].mean(), 2),
              'rho_tmdb': round(df['rating'].corr(df['tmdb_vote_average'], method='spearman'), 2),
              'rho_year': round(df['rating'].corr(df['year'], method='spearman'), 2)},
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text('window.DATA = ' + json.dumps(data, ensure_ascii=False) + ';\n', encoding='utf-8')
print('yazıldı:', OUT, '| film:', len(films), '| izleme listesi:', len(watch))
print('fazla sevdiklerim:', [f['n'] for f in above])
print('az sevdiklerim:', [f['n'] for f in below])


# ---------------------------------------------------------------------------
# site/catalog.js: predicted rating for every film in the TMDB catalog
# (run src/fetch_catalog.py first), plus my rated and watchlist films.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(ROOT / 'src'))
from features import build  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.base import clone  # noqa: E402

CAT_IN = PROC / 'tmdb_catalog.csv'
if CAT_IN.exists():
    rf = make_pipeline(SimpleImputer(strategy='median'),
                       RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                             random_state=42, n_jobs=1))
    Xr, yr = build(df, genre_cols), df['rating'].values
    # for films I rated: the prediction of a model that never saw that film
    oof = cross_val_predict(rf, Xr, yr, cv=KFold(5, shuffle=True, random_state=42))
    rf.fit(Xr, yr)

    cat = pd.read_csv(CAT_IN)
    enriched = pd.read_csv(PROC / 'films_enriched.csv')
    extra = enriched[~enriched['tmdb_id'].isin(cat['tmdb_id'])][
        ['tmdb_id', 'name', 'year', 'genres', 'original_language', 'tmdb_vote_average', 'tmdb_vote_count', 'popularity']]
    # Elif's films and a model of her ratings (src/build_together.py writes elif_enriched.csv;
    # her raw export is not in the repository, so this part is skipped without it)
    EL_R, EL_F = ROOT / 'data' / 'raw' / 'elif' / 'ratings.csv', PROC / 'elif_enriched.csv'
    has_elif = EL_R.exists() and EL_F.exists()
    if has_elif:
        ef = pd.read_csv(EL_F)
        ef['year'] = pd.to_numeric(ef['year'], errors='coerce')
        ef = ef[ef['year'].notna() & (ef['tmdb_vote_count'] > 0)]
        el = ef.merge(pd.read_csv(EL_R)[['Letterboxd URI', 'Rating']], left_on='uri', right_on='Letterboxd URI')
        el = el.rename(columns={'Rating': 'rating'}).drop_duplicates('tmdb_id')
        rf_el = clone(rf)
        Xe, ye = build(el, genre_cols), el['rating'].values
        oof_el = dict(zip(el['tmdb_id'], np.clip(
            cross_val_predict(rf_el, Xe, ye, cv=KFold(5, shuffle=True, random_state=42)), 0.5, 5)))
        rf_el.fit(Xe, ye)
        extra = pd.concat([extra, ef[~ef['tmdb_id'].isin(cat['tmdb_id'])][extra.columns]])
    cat = pd.concat([cat, extra], ignore_index=True).drop_duplicates('tmdb_id')
    cat['runtime'] = np.nan
    cat['pred'] = np.clip(rf.predict(build(cat, genre_cols)), 0.5, 5)
    if has_elif:
        # for films Elif rated: the prediction of a model that never saw that film
        cat['pred_el'] = np.clip(rf_el.predict(build(cat, genre_cols)), 0.5, 5)
        cat['pred_el'] = [oof_el.get(i, p) for i, p in zip(cat['tmdb_id'], cat['pred_el'])]

    mine = df.assign(oof=np.clip(oof, 0.5, 5)).drop_duplicates('tmdb_id').set_index('tmdb_id')
    wl = pd.read_csv(ROOT / 'data' / 'raw' / 'watchlist.csv') if (ROOT / 'data' / 'raw' / 'watchlist.csv').exists() else None
    wl_ids = set(enriched.loc[enriched['uri'].isin(wl['Letterboxd URI']), 'tmdb_id']) if wl is not None else set()

    gnames = sorted({g for s in cat['genres'].fillna('') for g in s.split('|') if g})
    gidx = {g: i for i, g in enumerate(gnames)}
    cat = cat.sort_values('tmdb_vote_count', ascending=False)
    rows = []
    for r in cat.itertuples():
        m = mine.loc[r.tmdb_id] if r.tmdb_id in mine.index else None
        rows.append([
            r.name, int(r.year), round(float(r.pred), 2), round(float(r.tmdb_vote_average), 1),
            [gidx[g] for g in str(r.genres).split('|') if g in gidx],
            float(m['rating']) if m is not None else None,      # my rating
            round(float(m['oof']), 2) if m is not None else None,  # model's unseen prediction
            1 if r.tmdb_id in wl_ids else 0,                    # on my watchlist
        ] + ([round(float(r.pred_el), 2)] if has_elif else []))  # Elif's predicted rating
    cat_out = ROOT / 'site' / 'catalog.js'
    cat_out.write_text('window.CATALOG = ' + json.dumps({'g': gnames, 'f': rows}, ensure_ascii=False,
                                                       separators=(',', ':')) + ';\n', encoding='utf-8')
    print(f'catalog: {len(rows)} films ({sum(1 for x in rows if x[5] is not None)} rated by me, '
          f'{sum(x[7] for x in rows)} on watchlist), {cat_out.stat().st_size / 1024:.0f} KB')
