import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import mlflow
import mlflow.sklearn
import logging

# --- Setup logging ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/train_model.log"),
        logging.StreamHandler()
    ]
)

# --- Data loading ---
db_path = os.path.join("database", "recipe_data.db")
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT Title, Ingredients FROM recipes", conn)
conn.close()

# Remove empty ingredient rows
df = df[df["Ingredients"].notnull() & (df["Ingredients"] != "")]
logging.info(f"Loaded {len(df)} recipes.")

# --- Train-test split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# --- Hyperparameter grid ---
param_grid = {
    "tfidf__max_features": [500, 1000, 1500],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "nn__n_neighbors": [5, 10, 15],
    "nn__metric": ["cosine"],
}
search_space = [
    (mf, ngr, nn, mt)
    for mf in param_grid["tfidf__max_features"]
    for ngr in param_grid["tfidf__ngram_range"]
    for nn in param_grid["nn__n_neighbors"]
    for mt in param_grid["nn__metric"]
]

# --- MLflow setup ---
mlflow.set_tracking_uri("http://localhost:5000")   # Change as needed
mlflow.set_experiment("PantryPalette_Recipe_Recommendation")

best_score = -np.inf
best_params = None
best_vectorizer = None
best_model = None

for i, (max_features, ngram_range, n_neighbors, metric) in enumerate(search_space):
    run_name = f"TFIDF_{max_features}_NGRAM_{ngram_range}_K_{n_neighbors}_METRIC_{metric}"
    with mlflow.start_run(run_name=run_name):
        # --- Vectorization ---
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
        X_train = vectorizer.fit_transform(train_df["Ingredients"])
        X_test = vectorizer.transform(test_df["Ingredients"])

        # --- Model ---
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn_model.fit(X_train)

        # --- Evaluate: avg max cosine similarity for each test recipe ---
        sims = cosine_similarity(X_test, X_train)
        avg_sim = float(sims.max(axis=1).mean())

        # --- MLflow logging ---
        mlflow.log_params({
            "tfidf_max_features": max_features,
            "tfidf_ngram_range": ngram_range,
            "nn_n_neighbors": n_neighbors,
            "nn_metric": metric
        })
        mlflow.log_metric("test_avg_max_cosine_similarity", avg_sim)

        logging.info(f"[{run_name}] avg max sim: {avg_sim:.4f}")

        if avg_sim > best_score:
            best_score = avg_sim
            best_params = {
                "max_features": max_features,
                "ngram_range": ngram_range,
                "n_neighbors": n_neighbors,
                "metric": metric,
            }
            best_vectorizer = vectorizer
            best_model = nn_model
            mlflow.sklearn.log_model(vectorizer, "best_tfidf_vectorizer")
            mlflow.sklearn.log_model(nn_model, "best_nearest_neighbors_model")
            logging.info(f"New best model found! Score: {best_score:.4f}")

# --- Save best artifacts to disk for Streamlit app ---
os.makedirs("model", exist_ok=True)
joblib.dump(best_vectorizer, os.path.join("model", "tfidf_vectorizer.pkl"))
joblib.dump(best_model, os.path.join("model", "nearest_neighbors.pkl"))
logging.info(f"Best model artifacts saved in 'model/'. Best params: {best_params}")
print(f"Best parameters: {best_params}, Best score: {best_score:.4f}")