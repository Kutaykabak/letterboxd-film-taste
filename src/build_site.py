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
    decades.append({'label': '1980 öncesi' if d == 1970 else f"{d}'ler", 'mean': round(x.mean(), 2), 'n': int(len(x))})


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
    ['Temel çizgi (ortalama)', 0.910, 0.070], ['Sadece TMDB puanı', 0.703, 0.057],
    ['Ridge', 0.686, 0.047], ['Spline + Ridge', 0.685, 0.053], ['Film yılı + TMDB puanı', 0.682, 0.054],
    ['Lasso', 0.671, 0.051], ['XGBoost', 0.665, 0.056], ['Gradient Boosting', 0.664, 0.056],
    ['LightGBM', 0.662, 0.058], ['Sıralı lojistik', 0.656, 0.056], ['Random Forest', 0.649, 0.054],
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
