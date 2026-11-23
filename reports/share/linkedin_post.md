# Sharing the project

Images in this folder (regenerate with `reports/share/render_share.ps1`):

| File | Size | Use |
|---|---|---|
| `linkedin-post.png` | 2160×2700 | Single-image LinkedIn post (4:5, fills the mobile feed) |
| `model-vs-me.png` | 2160×2700 | Second image, or a follow-up post |
| `linkedin-carousel.pdf` | 8 pages, 1080×1350 | LinkedIn "document" post (swipeable carousel) |
| `og.png`, `og-tr.png` | 1200×630 | Link preview cards, already wired into the site |
| `../powerbi/en-1-overview.png` (and pages 2, 3) | 2767×1600 | Power BI report pages, good as an extra image in a post |

A carousel is posted on LinkedIn with **Start a post → More (…) → Add a document**, then the PDF, then a document title (for example "What 355 film ratings say about my taste"). Put the link in the post text; LinkedIn shows it under "see more".

---

## Option A: single image post (`linkedin-post.png`)

I rated 355 films on Letterboxd over the last few years, so I turned my own ratings into a data project.

I joined them with TMDB data (genre, release year, runtime, language, crowd rating) and asked two questions: what do I actually like, and can a model predict my rating for a film I haven't seen?

A few things I found:

- The older the film, the higher I rate it. Before 1980 I average 4.4 stars, for the 2020s 3.0. Some of that is selection: from older decades I mostly watch films that are already well regarded.
- I agree with the crowd more than I expected (Spearman ρ = 0.60 with TMDB's average), with a few loud exceptions.
- My favourite result: Random Forest, XGBoost and LightGBM were not significantly better than a two-variable linear model (release year + TMDB rating) once I corrected for multiple comparisons. With 355 films, the honest answer was the simple one.

The model is off by about half a star on films it never saw (RMSE 0.66 with nested cross-validation, 0.91 for always guessing my average). I also rebuilt the analysis as a Power BI report with a star-schema model and DAX measures.

You can search my ratings and get a prediction for 5,000 popular films here: https://kutaykabak.github.io/letterboxd-film-taste/
Code and notebooks: https://github.com/Kutaykabak/letterboxd-film-taste

#datascience #statistics #machinelearning #python #letterboxd

---

## Option B: carousel post (`linkedin-carousel.pdf`)

Can a model learn my taste in films?

I took the 355 films I've rated on Letterboxd, added TMDB data, and tested what actually drives my ratings. Then I tried to predict them.

The short version is in the slides. The part I'm proudest of is slide 5: after a proper comparison (repeated and nested cross-validation, a held-out test set, multiple-comparison corrections), the fancy models didn't beat a simple two-variable rule. That's a less exciting result than "my Random Forest is 95% accurate", but it's the true one. Slide 7 is the same data as a Power BI report.

Try it on any film: https://kutaykabak.github.io/letterboxd-film-taste/
Code: https://github.com/Kutaykabak/letterboxd-film-taste

#datascience #statistics #machinelearning #python

---

## Türkçe sürüm (tek görsel)

Son birkaç yılda Letterboxd'da 355 film puanladım ve bu puanları bir veri projesine çevirdim.

Puanlarımı TMDB verisiyle (tür, yıl, süre, dil, izleyici puanı) birleştirdim ve iki soru sordum: gerçekte neyi seviyorum, ve bir model izlemediğim bir filme kaç puan vereceğimi tahmin edebilir mi?

Çıkan birkaç sonuç:

- Film ne kadar eskiyse o kadar yüksek puan veriyorum. 1980 öncesi filmlere ortalama 4,4, 2020'lerdekilere 3,0. Bunun bir kısmı seçilim: eski filmlerden zaten beğenilmiş olanları izliyorum.
- Kalabalıkla beklediğimden daha çok aynı fikirdeyim (TMDB ortalamasıyla Spearman ρ = 0,60), birkaç sert istisna dışında.
- En sevdiğim sonuç: Random Forest, XGBoost ve LightGBM, çoklu karşılaştırma düzeltmesinden sonra iki değişkenli basit bir doğrusal modelden (film yılı + TMDB puanı) anlamlı biçimde iyi çıkmadı. 355 filmde dürüst cevap basit olandı.

Model, hiç görmediği filmlerde ortalama yarım puan civarında yanılıyor (iç içe çapraz doğrulamayla RMSE 0,66; hep ortalamamı tahmin etmek 0,91). Aynı analizi yıldız şema veri modeli ve DAX ölçüleriyle bir Power BI raporu olarak da kurdum.

Puanlarımı arayabilir ve 5.000 popüler film için tahmin alabilirsiniz: https://kutaykabak.github.io/letterboxd-film-taste/tr/
Kod ve not defterleri: https://github.com/Kutaykabak/letterboxd-film-taste

#veribilimi #istatistik #makineöğrenmesi #python

---

## CV bullets

**Letterboxd film taste: analysis and rating prediction** (Python, scikit-learn, statsmodels, Power BI). [kutaykabak.github.io/letterboxd-film-taste](https://kutaykabak.github.io/letterboxd-film-taste/)

- Built an end-to-end pipeline on 355 personal film ratings enriched through the TMDB API; tested rating differences by decade, genre and language with non-parametric tests, bootstrap intervals and Benjamini-Hochberg correction.
- Compared 11 models (linear, spline, ordinal logistic, Random Forest, XGBoost, LightGBM) with repeated and nested cross-validation and an 80/20 hold-out; reduced RMSE from 0.91 to 0.66 and showed with corrected resampled t-tests that a two-variable model is statistically on par with tree ensembles.
- Published the results as an interactive bilingual web page that predicts ratings for 5,000 films, a Streamlit dashboard and a Power BI report with a star-schema model and DAX measures.

Shorter, one line: *Predicted my own film ratings from TMDB data (RMSE 0.91 → 0.66, nested CV), with statistical tests, an interactive web page and a Power BI report.*
