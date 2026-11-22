"""Letterboxd film zevki panosu.

Çalıştırma (proje kökünden):
    py -m streamlit run app\\streamlit_app.py
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from features import build  # noqa: E402

PROC = ROOT / 'data' / 'processed'
FIG = ROOT / 'reports' / 'figures'
NESTED_RMSE = 0.66          # 03_modelleme.ipynb, iç içe çapraz doğrulama
BLUE = '#4c78a8'

st.set_page_config(page_title='Letterboxd zevk analizi', page_icon='🎬', layout='wide')


# ---------- veri ve model ----------
@st.cache_data
def load():
    df = pd.read_csv(PROC / 'analysis_table.csv', parse_dates=['rated_date'])
    films = pd.read_csv(PROC / 'films_enriched.csv')
    wl_path = ROOT / 'data' / 'raw' / 'watchlist.csv'
    wl_uris = set(pd.read_csv(wl_path)['Letterboxd URI']) if wl_path.exists() else set()
    genres = [c for c in df.columns if c.startswith('g_')]
    return df, films, wl_uris, genres


@st.cache_resource
def train(_df, genres):
    model = make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                              random_state=42, n_jobs=1))
    model.fit(build(_df, genres), _df['rating'].values)
    return model


def predict(model, frame, genres):
    return np.clip(model.predict(build(frame, genres)), 0.5, 5)


df, films, wl_uris, genres = load()
model = train(df, genres)
genre_names = [g[2:] for g in genres]

unrated = films[~films['uri'].isin(set(df['uri']))].copy()
unrated['tahmin'] = predict(model, unrated, genres)
unrated['izleme_listesinde'] = unrated['uri'].isin(wl_uris)

# ---------- başlık ----------
st.title('Letterboxd film zevkim')
st.caption('355 filme verdiğim puanlar, TMDB verisiyle birleştirildi. '
           'Bu ürün TMDB API\'sini kullanır ancak TMDB tarafından onaylanmamıştır.')

c1, c2, c3, c4 = st.columns(4)
c1.metric('Puanlı film', f'{len(df)}')
c2.metric('Ortalama puanım', f'{df["rating"].mean():.2f}')
c3.metric('TMDB ile korelasyon (Spearman)', f'{df["rating"].corr(df["tmdb_vote_average"], method="spearman"):.2f}')
c4.metric('Tahmin hatası (RMSE)', f'{NESTED_RMSE:.2f}', help='İç içe çapraz doğrulama. Temel çizgi 0,91.')

tab1, tab2, tab3, tab4 = st.tabs(['Zevkim', 'Tahmin et', 'İzleme listesi', 'Model'])

# ---------- 1. Zevkim ----------
with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader('Puan dağılımı')
        dist = df['rating'].value_counts().sort_index().reset_index()
        dist.columns = ['puan', 'film']
        st.altair_chart(
            alt.Chart(dist).mark_bar(color=BLUE).encode(
                x=alt.X('puan:O', title='Verdiğim puan'),
                y=alt.Y('film:Q', title='Film sayısı'),
                tooltip=['puan', 'film']),
            use_container_width=True)

        st.subheader('On yıllara göre')
        dec = df.assign(on_yil=np.where(df['decade'] < 1980, '<1980', df['decade'].astype(str) + "'ler"))
        dec = dec.groupby('on_yil').agg(ortalama=('rating', 'mean'), film=('rating', 'size')).reset_index()
        st.altair_chart(
            alt.Chart(dec).mark_bar(color=BLUE).encode(
                x=alt.X('on_yil:N', title='Filmin on yılı', sort=None),
                y=alt.Y('ortalama:Q', title='Ortalama puan', scale=alt.Scale(domain=[0, 5])),
                tooltip=['on_yil', alt.Tooltip('ortalama:Q', format='.2f'), 'film']),
            use_container_width=True)

    with right:
        st.subheader('Türlere göre')
        min_n = st.slider('En az kaç film?', 3, 40, 15, key='min_n')
        rows = [{'tür': g[2:], 'ortalama': df.loc[df[g] == 1, 'rating'].mean(), 'film': int(df[g].sum())}
                for g in genres if df[g].sum() >= min_n]
        gen = pd.DataFrame(rows).sort_values('ortalama', ascending=False)
        overall = df['rating'].mean()
        bars = alt.Chart(gen).mark_bar().encode(
            x=alt.X('ortalama:Q', title='Ortalama puan', scale=alt.Scale(domain=[0, 5])),
            y=alt.Y('tür:N', sort='-x', title=None),
            color=alt.condition(alt.datum.ortalama >= overall, alt.value(BLUE), alt.value('#e45756')),
            tooltip=['tür', alt.Tooltip('ortalama:Q', format='.2f'), 'film'])
        rule = alt.Chart(pd.DataFrame({'x': [overall]})).mark_rule(strokeDash=[4, 4], color='gray').encode(x='x:Q')
        st.altair_chart(bars + rule, use_container_width=True)
        st.caption('Kesikli çizgi genel ortalamam. Mavi: ortalamanın üstü, kırmızı: altı.')

    st.subheader('Benim puanım ve TMDB puanı')
    sc = df[['name', 'year', 'rating', 'tmdb_vote_average', 'decade']].copy()
    sc['on_yil'] = np.where(sc['decade'] < 1980, '<1980', sc['decade'].astype(str))
    st.altair_chart(
        alt.Chart(sc).mark_circle(size=55, opacity=0.65).encode(
            x=alt.X('tmdb_vote_average:Q', title='TMDB ortalama puanı', scale=alt.Scale(zero=False)),
            y=alt.Y('rating:Q', title='Verdiğim puan'),
            color=alt.Color('on_yil:N', title='On yıl'),
            tooltip=['name', 'year', 'rating', 'tmdb_vote_average']).interactive(),
        use_container_width=True)

    st.subheader('Filmlerim')
    q = st.text_input('Film ara', placeholder='örn. Bergman, Se7en')
    show = df[['name', 'year', 'rating', 'tmdb_vote_average', 'genres', 'directors', 'liked']]
    if q:
        mask = show['name'].str.contains(q, case=False, na=False) | show['directors'].str.contains(q, case=False, na=False)
        show = show[mask]
    st.dataframe(show.sort_values('rating', ascending=False), hide_index=True, use_container_width=True,
                 column_config={'name': 'Film', 'year': st.column_config.NumberColumn('Yıl', format='%d'),
                                'rating': 'Puanım', 'tmdb_vote_average': 'TMDB', 'genres': 'Türler',
                                'directors': 'Yönetmen', 'liked': st.column_config.CheckboxColumn('Kalp')})

# ---------- 2. Tahmin et ----------
with tab2:
    st.write('Model, filmin TMDB bilgilerinden kaç puan vereceğimi tahmin ediyor. '
             f'Ortalama hata yaklaşık ±{NESTED_RMSE:.2f} puan, o yüzden tahmini bir yön göstergesi olarak okumak lazım.')
    mode = st.radio('Nasıl?', ['Listeden film seç', 'Kendi filmini tanımla'], horizontal=True)

    if mode == 'Listeden film seç':
        opts = unrated.sort_values('name')
        label = opts['name'] + ' (' + opts['year'].astype(str) + ')'
        pick = st.selectbox('İzlediğim ama puanlamadığım ya da izleme listemdeki filmler', label.tolist())
        row = opts[label == pick].iloc[0]
        a, b = st.columns([1, 2])
        a.metric('Tahmini puanım', f'{row["tahmin"]:.1f} / 5')
        b.markdown(f'**{row["name"]}** ({row["year"]})  \n'
                   f'Yönetmen: {row["directors"] or "?"}  \n'
                   f'Türler: {str(row["genres"]).replace("|", ", ")}  \n'
                   f'TMDB: {row["tmdb_vote_average"]:.1f} ({int(row["tmdb_vote_count"])} oy)')
    else:
        a, b = st.columns(2)
        year = a.slider('Yıl', 1920, 2026, 2010)
        tmdb = a.slider('TMDB puanı', 4.0, 9.0, 7.5, 0.1)
        votes = a.select_slider('TMDB oy sayısı', [100, 500, 1000, 3000, 8000, 15000, 30000], value=8000)
        runtime = b.slider('Süre (dk)', 60, 240, 120)
        lang = b.selectbox('Dil', ['İngilizce', 'Türkçe', 'Diğer'])
        gsel = b.multiselect('Türler', genre_names, default=['Drama'])
        frame = pd.DataFrame([{
            'genres': '|'.join(gsel), 'year': year, 'runtime': runtime, 'tmdb_vote_average': tmdb,
            'tmdb_vote_count': votes, 'popularity': films['popularity'].median(),
            'original_language': {'İngilizce': 'en', 'Türkçe': 'tr', 'Diğer': 'xx'}[lang]}])
        st.metric('Tahmini puanım', f'{predict(model, frame, genres)[0]:.1f} / 5')
        st.caption('Popülerlik, eğitim verisinin ortancasına sabitlendi.')

# ---------- 3. İzleme listesi ----------
with tab3:
    st.write('İzleme listemdeki filmler, modelin tahmin ettiği puana göre sıralı.')
    wl = unrated[unrated['izleme_listesinde']].copy()
    a, b = st.columns(2)
    gf = a.multiselect('Tür filtresi (hepsini içersin)', genre_names)
    yr = b.slider('Yıl aralığı', int(wl['year'].min()), int(wl['year'].max()),
                  (int(wl['year'].min()), int(wl['year'].max())))
    for g in gf:
        wl = wl[wl['genres'].fillna('').str.split('|').apply(lambda xs, g=g: g in xs)]
    wl = wl[wl['year'].between(*yr)].sort_values('tahmin', ascending=False)
    st.caption(f'{len(wl)} film')
    st.dataframe(wl[['name', 'year', 'tahmin', 'tmdb_vote_average', 'genres', 'directors']],
                 hide_index=True, use_container_width=True,
                 column_config={'name': 'Film', 'year': st.column_config.NumberColumn('Yıl', format='%d'),
                                'tahmin': st.column_config.ProgressColumn('Tahmin', min_value=0.5, max_value=5,
                                                                          format='%.2f'),
                                'tmdb_vote_average': 'TMDB', 'genres': 'Türler', 'directors': 'Yönetmen'})

# ---------- 4. Model ----------
with tab4:
    st.markdown(f'''
Burada kullanılan model **Random Forest**. 11 modeli 5 katlı, 10 tekrarlı çapraz doğrulamayla karşılaştırdım.
Model seçimini de hesaba katan iç içe çapraz doğrulamada hata **RMSE {NESTED_RMSE:.2f}**, ortalamayı tahmin eden
temel çizgide 0,91. Random Forest, XGBoost, LightGBM ve sıralı lojistik regresyon birbirine çok yakın çıktı.
Hiçbiri "film yılı + TMDB puanı" ile kurulan basit modeli çoklu karşılaştırma düzeltmesinden sonra anlamlı biçimde geçmedi.

Ayrıntılar `notebooks/03_modelleme.ipynb` içinde.
''')
    for f, cap in [('06_model_karsilastirma.png', 'Model karşılaştırması'),
                   ('07_tahmin_vs_gercek.png', 'Görmediği filmlerde tahmin ve gerçek puan'),
                   ('08_ozellik_onemi.png', 'Permütasyon önemi')]:
        if (FIG / f).exists():
            st.image(str(FIG / f), caption=cap, width=620)
