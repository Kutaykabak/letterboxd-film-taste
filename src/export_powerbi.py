"""Power BI için tabloları data/powerbi/ altına yazar.

Kullanım (proje kökünden):
    py src\\export_powerbi.py

Tablolar (yıldız şema):
    films.csv         olgu tablosu, puanladığım her film bir satır (film_id anahtar)
    film_genres.csv   köprü tablo: film_id, genre (bir filmin birden fazla türü olabilir)
    genres.csv        tür boyutu: genre, genre_tr
    watchlist.csv     izleme listemdeki filmler ve modelin tahmini
    model_results.csv modellerin hata ölçüleri (notebooks/03_modelleme.ipynb sonuçları)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from features import build

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
OUT = ROOT / 'data' / 'powerbi'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

GENRE_TR = {'Drama': 'Dram', 'Crime': 'Suç', 'Comedy': 'Komedi', 'Horror': 'Korku',
            'Science Fiction': 'Bilim kurgu', 'Animation': 'Animasyon', 'Romance': 'Romantik',
            'Thriller': 'Gerilim', 'War': 'Savaş', 'History': 'Tarih', 'Mystery': 'Gizem',
            'Fantasy': 'Fantastik', 'Adventure': 'Macera', 'Action': 'Aksiyon', 'Family': 'Aile',
            'Music': 'Müzik', 'Documentary': 'Belgesel', 'Western': 'Western', 'TV Movie': 'TV filmi'}


def first(s):
    return s.fillna('').str.split('|').str[0].replace('', np.nan)


def write(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    print(f'{name:18s} {len(df):4d} satır, {df.shape[1]:2d} sütun')


df = pd.read_csv(PROC / 'analysis_table.csv', parse_dates=['rated_date'])
genres = [c for c in df.columns if c.startswith('g_')]
df['film_id'] = np.arange(1, len(df) + 1)

# modelin her filmi görmeden yaptığı tahmin (5 katlı CV, not defterindeki Random Forest ayarları)
rf = make_pipeline(SimpleImputer(strategy='median'),
                   RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                         random_state=SEED, n_jobs=1))
X = build(df, genres)
oof = cross_val_predict(rf, X, df['rating'].values, cv=KFold(5, shuffle=True, random_state=SEED))

# sadece TMDB puanından beklenen puan (doğrusal uyum)
b, a = np.polyfit(df['tmdb_vote_average'], df['rating'], 1)

lang = np.select([df['original_language'] == 'en', df['original_language'] == 'tr'], ['English', 'Turkish'], 'Other')
films = pd.DataFrame({
    'film_id': df['film_id'],
    'name': df['name'],
    'year': df['year'],
    'decade': df['decade'],
    # alfabetik sıra kronolojik sırayla aynı olsun diye: "1970 ve öncesi" < "1980'ler" < ...
    'decade_label': np.where(df['decade'] < 1980, '1970 ve öncesi', df['decade'].astype(str) + "'ler"),
    'decade_label_en': np.where(df['decade'] < 1980, '1970s and earlier', df['decade'].astype(str) + 's'),
    'rating': df['rating'],
    'loved': df['loved'],
    'liked': df['liked'],
    'rated_date': df['rated_date'].dt.date,
    'rated_year': df['rated_year'],
    'tmdb_rating': df['tmdb_vote_average'].round(2),
    'tmdb_votes': df['tmdb_vote_count'].astype(int),
    'popularity': df['popularity'].round(2),
    'runtime': df['runtime'],
    'original_language': df['original_language'],
    'language_group': lang,
    'primary_country': df['primary_country'],
    'director': df['director'],
    'n_genres': df['n_genres'],
    'genres': df['genres'],
    'expected_from_tmdb': (a + b * df['tmdb_vote_average']).round(2),
    'predicted_rating': np.clip(oof, 0.5, 5).round(2),
})
films['diff_from_crowd'] = (films['rating'] - films['expected_from_tmdb']).round(2)
films['prediction_error'] = (films['predicted_rating'] - films['rating']).round(2)
films['abs_error'] = films['prediction_error'].abs()
write(films, 'films.csv')

fg = (df[['film_id', 'genres']].assign(genre=df['genres'].fillna('').str.split('|'))
      .explode('genre').query("genre != ''")[['film_id', 'genre']])
write(fg, 'film_genres.csv')

g = pd.DataFrame({'genre': sorted(fg['genre'].unique())})
g['genre_tr'] = g['genre'].map(GENRE_TR).fillna(g['genre'])
write(g, 'genres.csv')

wl = pd.read_csv(ROOT / 'reports' / 'izleme_listesi_tahminleri.csv')
wl = wl.rename(columns={'tahmin': 'predicted_rating', 'tmdb_vote_average': 'tmdb_rating', 'directors': 'director'})
wl['director'] = first(wl['director'])
wl.insert(0, 'rank', np.arange(1, len(wl) + 1))
write(wl, 'watchlist.csv')

# notebooks/03_modelleme.ipynb: 5x10 tekrarlı CV, iç içe CV ve 80/20 tutma testi
models = pd.DataFrame([
    ['Baseline (mean)', 'Baseline', 0.910, 0.070, 0.738, -0.014],
    ['TMDB rating only', 'Linear', 0.703, 0.057, 0.553, 0.386],
    ['Release year + TMDB rating', 'Linear', 0.682, 0.054, 0.533, 0.423],
    ['Ridge (all features)', 'Linear', 0.686, 0.047, 0.525, 0.418],
    ['Lasso (all features)', 'Linear', 0.671, 0.051, 0.513, 0.443],
    ['Spline + Ridge', 'Linear', 0.685, 0.053, 0.533, 0.419],
    ['Ordinal logistic', 'Ordinal', 0.656, 0.056, 0.509, 0.467],
    ['Random Forest', 'Tree ensemble', 0.649, 0.054, 0.511, 0.479],
    ['Gradient Boosting', 'Tree ensemble', 0.664, 0.056, 0.517, 0.453],
    ['XGBoost', 'Tree ensemble', 0.665, 0.056, 0.519, 0.451],
    ['LightGBM', 'Tree ensemble', 0.662, 0.058, 0.519, 0.456],
], columns=['model', 'family', 'rmse', 'rmse_sd', 'mae', 'r2'])
models['evaluation'] = 'Repeated 5-fold CV (5x10)'
extra = pd.DataFrame([
    ['Nested CV (selection included)', 'Selection', 0.662, 0.097, np.nan, np.nan, 'Nested CV (outer 5x2)'],
    ['Random Forest, 80/20 hold-out', 'Tree ensemble', 0.666, np.nan, 0.520, 0.396, '80/20 hold-out test (n=71)'],
    ['Baseline, 80/20 hold-out', 'Baseline', 0.857, np.nan, 0.717, -0.001, '80/20 hold-out test (n=71)'],
], columns=list(models.columns))
write(pd.concat([models, extra], ignore_index=True), 'model_results.csv')

print('\nModelin görmeden tahmini: RMSE', round(float(np.sqrt(np.mean((oof - df['rating']) ** 2))), 3))
print('Klasör:', OUT)
