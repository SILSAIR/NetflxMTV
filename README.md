# NetflxMTV

Project: Predicting Netflix Content Rating from Metadata
Type: Classification
Target variable: rating_group (Kids / Teens / Mature, grouped from original rating column)
Dataset: netflix_titles.csv, original 8807 rows x 12 columns, final 8790 rows x 12 columns


Glitch rows
------------
Found 3 rows with duration values ("74 min", "84 min", "66 min") in the rating column instead of an actual rating, caused by a column-shift error. Removed.


Missing values
---------------
df.isnull().sum() before cleaning:

    director        2631
    cast             825
    country          830
    date_added        10
    rating             4  (after removing glitch rows)

Dropped rows missing rating (target variable). Filled director, cast, country with "Unknown" (too many missing to drop). Dropped 10 rows missing date_added.

df.isnull().sum() after cleaning:

    show_id         0
    type            0
    title           0
    director        0
    cast            0
    country         0
    date_added      0
    release_year    0
    rating          0
    duration        0
    listed_in       0
    description     0
    rating_group    0


Duplicates
-----------
df.duplicated().sum():

    Duplicate rows: 0

No action needed.


Data types
-----------
Converted date_added from text to datetime. Extracted numeric part of duration into duration_num.

df.dtypes confirmed date_added as datetime64[us].


Country cleaning
------------------
Split multi-country cells (e.g. "United States, India") and kept only the primary country.

df['country'].value_counts().head(10):

    country
    United States     3202
    India              1008
    Unknown             829
    United Kingdom      627
    Canada               271
    Japan                257
    France               212
    South Korea          211
    Spain                181
    Mexico               134


Outliers / invalid values
---------------------------
describe() on release_year and duration_num:

    release_year: min 1925, max 2021, mean ~2014
    duration_num: min 1.0, max 312.0, mean ~69.9, std ~50.8

release_year: no invalid values. duration_num mixed two different units (minutes for Movies, seasons for TV Shows), making the stats above misleading. Split into two columns:

    duration_minutes (Movies only): count 6126, mean ~99.6, min 3, max 312
    duration_seasons (TV Shows only): count 2664, mean ~1.75, min 1, max 17


Dropped columns
-----------------
Dropped show_id (ID only), description (free text, unused), duration and duration_num (replaced by duration_minutes / duration_seasons).

df.columns.tolist() and df.shape:

    ['type', 'title', 'director', 'cast', 'country', 'date_added',
     'release_year', 'rating', 'listed_in', 'rating_group',
     'duration_minutes', 'duration_seasons']
    (8790, 12)
