"""Model özelliklerini üreten ortak fonksiyon.

03_modelleme.ipynb içindeki build() ile aynı mantık. Streamlit uygulaması bunu kullanıyor.
"""
import numpy as np
import pandas as pd


def build(frame: pd.DataFrame, genres: list[str]) -> pd.DataFrame:
    """TMDB sütunlarından (genres, year, runtime, tmdb_vote_average, tmdb_vote_count,
    popularity, original_language) model özelliklerini üretir.

    genres: eğitim tablosundaki tür sütunları ('g_Drama' gibi). Tüm tablolarda aynı
    sütunlar olsun diye eksik türler 0 ile doldurulur.
    """
    g = (frame['genres'].fillna('').str.get_dummies(sep='|').add_prefix('g_')
         .reindex(columns=genres, fill_value=0))
    X = pd.DataFrame({
        'year': frame['year'],
        'runtime': frame['runtime'].replace(0, np.nan),
        'tmdb_vote_average': frame['tmdb_vote_average'],
        'log_votes': np.log1p(frame['tmdb_vote_count']),
        'log_popularity': np.log1p(frame['popularity']),
        'n_genres': g.sum(axis=1),
        'is_english': (frame['original_language'] == 'en').astype(int),
        'is_turkish': (frame['original_language'] == 'tr').astype(int),
    })
    out = pd.concat([X, g], axis=1)
    out.columns = out.columns.str.replace(' ', '_')
    return out
