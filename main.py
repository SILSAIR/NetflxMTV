import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
df = df.drop(columns=['show_id', 'description', 'duration', 'duration_num'])
