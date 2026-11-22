// Power BI Desktop: Veri al > Boş sorgu > Gelişmiş Düzenleyici, her sorguyu ayrı yapıştır.
// Sayılar "en-US" kültürüyle okunuyor, böylece Türkçe Windows'ta 4.5 sayısı 45 olmuyor.
// Klasör yolu değişirse sadece Folder değerini düzelt.

// ---- films ----
let
    Folder = "D:\claude\letterboxd-projesi\data\powerbi\",
    Source = Csv.Document(File.Contents(Folder & "films.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Headers, {
        {"film_id", Int64.Type}, {"name", type text}, {"year", Int64.Type}, {"decade", Int64.Type},
        {"decade_label", type text}, {"decade_label_en", type text}, {"rating", type number}, {"loved", Int64.Type}, {"liked", Int64.Type},
        {"rated_date", type date}, {"rated_year", Int64.Type}, {"tmdb_rating", type number},
        {"tmdb_votes", Int64.Type}, {"popularity", type number}, {"runtime", type number},
        {"original_language", type text}, {"language_group", type text}, {"primary_country", type text},
        {"director", type text}, {"n_genres", Int64.Type}, {"genres", type text},
        {"expected_from_tmdb", type number}, {"predicted_rating", type number},
        {"diff_from_crowd", type number}, {"prediction_error", type number}, {"abs_error", type number}
    }, "en-US")
in
    Typed

// ---- film_genres ----
let
    Folder = "D:\claude\letterboxd-projesi\data\powerbi\",
    Source = Csv.Document(File.Contents(Folder & "film_genres.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Headers, {{"film_id", Int64.Type}, {"genre", type text}}, "en-US")
in
    Typed

// ---- genres ----
let
    Folder = "D:\claude\letterboxd-projesi\data\powerbi\",
    Source = Csv.Document(File.Contents(Folder & "genres.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Headers, {{"genre", type text}, {"genre_tr", type text}}, "en-US")
in
    Typed

// ---- watchlist ----
let
    Folder = "D:\claude\letterboxd-projesi\data\powerbi\",
    Source = Csv.Document(File.Contents(Folder & "watchlist.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Headers, {
        {"rank", Int64.Type}, {"name", type text}, {"year", Int64.Type}, {"predicted_rating", type number},
        {"genres", type text}, {"director", type text}, {"tmdb_rating", type number}
    }, "en-US")
in
    Typed

// ---- model_results ----
let
    Folder = "D:\claude\letterboxd-projesi\data\powerbi\",
    Source = Csv.Document(File.Contents(Folder & "model_results.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Headers, {
        {"model", type text}, {"family", type text}, {"rmse", type number}, {"rmse_sd", type number},
        {"mae", type number}, {"r2", type number}, {"evaluation", type text}
    }, "en-US")
in
    Typed
