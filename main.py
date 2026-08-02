import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
	confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('netflix_titles.csv')


df = df[~df['rating'].isin(['74 min', '84 min', '66 min'])]
df = df.dropna(subset=['rating'])
rating_map = {
    'G': 'Kids', 'TV-G': 'Kids', 'TV-Y': 'Kids', 'TV-Y7': 'Kids', 'TV-Y7-FV': 'Kids',
    'PG': 'Teens', 'PG-13': 'Teens', 'TV-PG': 'Teens', 'TV-14': 'Teens',
    'R': 'Mature', 'NC-17': 'Mature', 'TV-MA': 'Mature', 'NR': 'Mature', 'UR': 'Mature'
}
df['rating_group'] = df['rating'].map(rating_map)


df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

df = df.dropna(subset=['date_added'])
df['date_added'] = pd.to_datetime(df['date_added'].str.strip())

df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)

df['country'] = df['country'].str.split(',').str[0].str.strip()

df['duration_minutes'] = df.apply(lambda row: row['duration_num'] if row['type'] == 'Movie' else None, axis=1)
df['duration_seasons'] = df.apply(lambda row: row['duration_num'] if row['type'] == 'TV Show' else None, axis=1)
df['duration_minutes'] = df['duration_minutes'].fillna(0)
df['duration_seasons'] = df['duration_seasons'].fillna(0)

df = df.drop(columns=['show_id', 'description', 'duration', 'duration_num'])


df['primary_genre'] = df['listed_in'].str.split(',').str[0].str.strip()

top_countries = df['country'].value_counts().head(10).index
df['country_grouped'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')
top_genres = df['primary_genre'].value_counts().head(10).index
df['genre_grouped'] = df['primary_genre'].apply(lambda x: x if x in top_genres else 'Other')

le_type = LabelEncoder()
le_country = LabelEncoder()
le_genre = LabelEncoder()
le_target = LabelEncoder()
df['type_enc'] = le_type.fit_transform(df['type'])
df['country_enc'] = le_country.fit_transform(df['country_grouped'])
df['genre_enc'] = le_genre.fit_transform(df['genre_grouped'])
df['rating_group_enc'] = le_target.fit_transform(df['rating_group'])

features = ['type_enc', 'duration_minutes', 'duration_seasons', 'release_year', 'country_enc', 'genre_enc']
X = df[features]
y = df['rating_group_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(
    n_neighbors=5
)
model.fit(
    X_train_scaled,
    y_train
)
predictions = model.predict(
    X_test_scaled
)
results = pd.DataFrame({
    "Actual Species": y_test.reset_index(drop=True),
    "Predicted Species": predictions
})

results["Correct Prediction"] = (
    results["Actual Species"] ==
    results["Predicted Species"]
)

cm = confusion_matrix(
    y_test,
    predictions
)
accuracy = accuracy_score(
    y_test,
    predictions
)
precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


comparison = results.head(20).copy()

comparison["Sample"] = range(
    1,
    len(comparison) + 1
)
matrix_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

matrix_display.plot(
    cmap="Greens"
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


comparison_melted = comparison.melt(
    id_vars="Sample",
    value_vars=[
        "Actual Species",
        "Predicted Species"
    ],
    var_name="Value Type",
    value_name="Species"
)

plt.figure(figsize=(11, 6))

sns.scatterplot(
    data=comparison_melted,
    x="Sample",
    y="Species",
    hue="Value Type",
    style="Value Type",
    s=100
)

plt.title("Actual vs Predicted Species")
plt.xlabel("Test Sample")
plt.ylabel("Species")
plt.tight_layout()
plt.show()

print("\nFeature order:")
print(X.columns.tolist())

new_title = pd.DataFrame(
    [[0, 105, 0, 2023, 9, 5]],
    columns=X.columns
)

new_title_scaled = scaler.transform(new_title)
new_prediction = model.predict(new_title_scaled)

print("\n" + "=" * 45)
print("NetflxMTV KNN CLASSIFICATION RESULTS")
print("=" * 45)
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("=" * 45)
print("Project completed successfully.")