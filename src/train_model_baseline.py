import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# === Baseline Training Script ===

# Load cleaned recipe data
conn = sqlite3.connect("database/recipe_data.db")
df = pd.read_sql_query("SELECT Title, Ingredients FROM recipes", conn)
conn.close()

# Basic preprocessing: drop nulls
df = df[df["Ingredients"].notnull() & (df["Ingredients"] != "")]

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# TF-IDF vectorizer (simple config)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_train = vectorizer.fit_transform(train_df["Ingredients"])

# NearestNeighbors model
model = NearestNeighbors(n_neighbors=10, metric="cosine")
model.fit(X_train)

# Save vectorizer and model
os.makedirs("model", exist_ok=True)
joblib.dump(vectorizer, "model/tfidf_vectorizer_baseline.pkl")
joblib.dump(model, "model/nearest_neighbors_baseline.pkl")

print("Baseline TF-IDF + NearestNeighbors model saved.")
