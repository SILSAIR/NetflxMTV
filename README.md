# NetflxMTV

Project: Predicting Netflix Content Rating from Metadata<br>
Type: Classification<br>
Target variable: rating_group (Kids / Teens / Mature)<br>
Dataset: netflix_titles.csv, original 8807 rows x 12 columns


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
Key insights
-------------
1. Rating column had a 3-row column-shift glitch where duration values appeared instead of ratings.
2. Duration required splitting since movies and TV shows use different units (minutes vs. seasons).
3. Raw rating column had 15+ categories, several with fewer than 10 rows; grouped into 3 balanced classes (Kids, Teens, Mature) for classification.

Genre cleaning
---------------
listed_in contained multiple genres per row (e.g. "Comedies, Dramas, International"). Kept only the primary (first-listed) genre in a new column, primary_genre.
 
df['primary_genre'].value_counts().head(10):
 
    Dramas                      1599
    Comedies                    1210
    Action & Adventure           859
    Documentaries                829
    International TV Shows       773
    Children & Family Movies     605
    Crime TV Shows               399
    Kids' TV                     385
    Stand-Up Comedy              334
    Horror Movies                275
 
 
Grouping rare countries/genres into "Other"
---------------------------------------------
Kept top 10 countries and top 10 primary genres by frequency; grouped everything else into "Other" to avoid feeding 80+ sparse categories into the model.
 
df['country_grouped'].value_counts():
 
    United States     3202
    Other              1858
    India              1008
    Unknown             829
    United Kingdom      627
    Canada               271
    Japan                257
    France               212
    South Korea          211
    Spain                181
    Mexico               134
 
df['genre_grouped'].value_counts():
 
    Dramas                      1599
    Other                       1522
    Comedies                    1210
    Action & Adventure           859
    Documentaries                829
    International TV Shows       773
    Children & Family Movies     605
    Crime TV Shows               399
    Kids' TV                     385
    Stand-Up Comedy              334
    Horror Movies                275
 
"Other" is the 2nd largest group in both cases but not dominant, so top-10 cutoff was kept as-is.
 
 
Encoding categorical columns
------------------------------
Used sklearn's LabelEncoder to convert type, country_grouped, genre_grouped, and rating_group into numeric columns for modeling.
 
type -> type_enc:
    Movie     0
    TV Show   1
 
rating_group -> rating_group_enc:
    Kids     0
    Mature   1
    Teens    2
 
country_grouped -> country_enc:
    Canada             0
    France              1
    India                2
    Japan               3
    Mexico              4
    Other                5
    South Korea         6
    Spain                7
    United Kingdom       8
    United States        9
    Unknown              10
 
genre_grouped -> genre_enc:
    Action & Adventure          0
    Children & Family Movies    1
    Comedies                    2
    Crime TV Shows              3
    Documentaries                4
    Dramas                       5
    Horror Movies                6
    International TV Shows       7
    Kids' TV                     8
    Other                        9
    Stand-Up Comedy              10
 
 
Train/test split
------------------
Final features: type_enc, duration_minutes, duration_seasons, release_year, country_enc, genre_enc
Target: rating_group_enc
Split: 80% train / 20% test, stratified on target, random_state=42
 
Train size: (7032, 6)
Test size: (1758, 6)
 
 
Training the Decision Tree
-----------------------------
Model: DecisionTreeClassifier(max_depth=8, random_state=42)
Trained on X_train/y_train.
 
Training accuracy: 0.6879977246871445
Test accuracy: 0.6575654152445961
 
Train and test accuracy are close (no major overfitting). Both well above the 33% baseline for random guessing across 3 classes.
 
 
Trying KNN as a second model
-------------------------------
Filled remaining NaNs in duration_minutes and duration_seasons with 0 (KNN cannot handle missing values, unlike the Decision Tree).
 
Applied StandardScaler to features (fit on X_train only, applied to X_test) since KNN relies on distance and features were on very different scales (e.g. release_year in the 1900s-2000s vs binary type_enc).
 
Model: KNeighborsClassifier(n_neighbors=5)
 
Training accuracy: 0.7286689419795221
Test accuracy: 0.6143344709897611
 
Comparison:
    Decision Tree  - train 0.688, test 0.658
    KNN (k=5)      - train 0.729, test 0.614
 
KNN has a wider train/test gap (more overfitting) and lower test accuracy than the Decision Tree. Decision Tree currently performs better on unseen data.
 
 
Actual vs predicted sample (KNN)
-----------------------------------
Built a comparison table of actual vs predicted rating_group_enc values on the test set, with a Correct Prediction flag.
 
First 10 rows:
    Actual  Predicted  Correct
    2       2          True
    1       1          True
    2       0          False
    2       2          True
    1       1          True
    2       2          True
    1       1          True
    1       1          True
    1       1          True
    0       0          True
 
9 out of the first 10 predictions were correct, consistent with the overall ~61% test accuracy. The one miss (row 2) confused class 2 (Teens) with class 0 (Kids).
 
 
Confirming accuracy with accuracy_score
------------------------------------------
Accuracy: 0.6143
 
Matches the KNN test accuracy computed earlier via knn.score(X_test_scaled, y_test).
 
 
Precision (weighted)
-----------------------
Precision: 0.6132
 
Close to overall accuracy (0.6143), suggesting the model isn't wildly over- or under-predicting any one class relative to its actual frequency.
 
 
Recall (weighted)
---------------------
Recall: 0.6143
 
Matches accuracy almost exactly, consistent with precision - no major imbalance in how well each class is being identified.
 
 
F1 Score (weighted)
-----------------------
F1 Score: 0.6133
 
All four weighted metrics (accuracy 0.6143, precision 0.6132, recall 0.6143, F1 0.6133) are tightly clustered, indicating the KNN model's performance is consistent across metrics with no major precision/recall tradeoff issue.
 
 
Classification report (KNN, per class)
------------------------------------------
Class mapping: 0 = Kids, 1 = Mature, 2 = Teens
 
              precision    recall  f1-score   support
    Kids          0.67      0.66      0.67       181
    Mature        0.63      0.67      0.65       818
    Teens         0.58      0.55      0.56       759
    accuracy                          0.61      1758
    macro avg     0.63      0.63      0.63      1758
    weighted avg  0.61      0.61      0.61      1758
 
Kids is the best-performing class despite having the fewest samples (181). Teens is the weakest class (precision 0.58, recall 0.55), likely because Teens-rated content (PG, PG-13, TV-PG, TV-14) overlaps in genre and duration patterns with both Kids and Mature content, making it harder for the model to separate.
 
 
Confusion matrix (KNN)
--------------------------
Rows = actual, columns = predicted. Order: Kids, Mature, Teens.
 
                Predicted Kids  Predicted Mature  Predicted Teens
Actual Kids              120                22               39
Actual Mature              14               545              259
Actual Teens                44               300              415
 
Dominant error source is Mature <-> Teens confusion (259 Mature misclassified as Teens, 300 Teens misclassified as Mature = 559 total), consistent with the weaker precision/recall seen for the Teens class in the classification report. Kids is rarely confused with Mature (22 + 14 = 36), suggesting Kids content is more distinguishable from Mature than Teens content is.
 
 
Confusion matrix - visualized
---------------------------------
Plotted the same confusion matrix using ConfusionMatrixDisplay (Greens colormap) for the report/visualizations section.
 
Visual confirms: strong diagonal for class 1 (Mature, 545), with the darkest off-diagonal cells being (1,2)=259 and (2,1)=300 - visually reinforcing the Mature/Teens confusion as the model's main weak spot. Class 0 (Kids) has the lightest off-diagonal cells, confirming it's the most cleanly separated class.
 
 
Actual vs predicted - scatter comparison (first 20 test samples)
----------------------------------------------------------------------
Plotted actual vs predicted rating_group_enc values side by side for the first 20 test samples, using overlapping markers (circle = actual, X = predicted) to visually spot mismatches.
 
Out of 20 samples shown, only sample 3 shows a clear mismatch: actual = 2 (Teens), predicted = 0 (Kids). All other 19 samples show overlapping markers, meaning correct predictions. Consistent with the earlier actual-vs-predicted table and overall ~61% test accuracy.
 
 
Single new-title prediction test
------------------------------------
Tested the trained KNN model on a hypothetical new title (not in the dataset): a Movie, 105 minutes, released 2023, primary country United States, primary genre Dramas.
 
Feature order: ['type_enc', 'duration_minutes', 'duration_seasons', 'release_year', 'country_enc', 'genre_enc']
Input: [0, 105, 0, 2023, 9, 5]
 
Predicted rating group (encoded): 1
Predicted rating group (label): Mature
 
Result is plausible - a recent US-produced drama is realistically more likely to carry a Mature (TV-MA/R) rating than Kids or Teens, matching real-world patterns.
