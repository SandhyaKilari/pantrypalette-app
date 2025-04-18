import os
import pandas as pd
import sqlite3
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load cleaned ingredients from SQLite
conn = sqlite3.connect("../database/recipe_data.db")
df = pd.read_sql_query("SELECT Title, Ingredients FROM recipes", conn)
conn.close()

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Change if using remote tracking
mlflow.set_experiment("PantryPalette_Recipe_Recommendation")

with mlflow.start_run(run_name="tfidf_knn_training") as run:
    # 1) Log parameters
    params = {
        "tfidf_max_features": 500,
        "tfidf_ngram_range": (1, 2),
        "nn_n_neighbors": 10,
        "nn_metric": "cosine",
    }
    for k, v in params.items():
        mlflow.log_param(k, v)

    # 2) Vectorize
    vectorizer = TfidfVectorizer(
        max_features=params["tfidf_max_features"],
        ngram_range=params["tfidf_ngram_range"]
    )
    train_vectors = vectorizer.fit_transform(train_df["Ingredients"])

    # 3) Fit NearestNeighbors
    nn_model = NearestNeighbors(
        n_neighbors=params["nn_n_neighbors"],
        metric=params["nn_metric"]
    )
    nn_model.fit(train_vectors)

    # 4) Evaluate on test set using cosine similarity
    test_vectors = vectorizer.transform(test_df["Ingredients"])
    similarities = cosine_similarity(test_vectors, train_vectors)
    avg_sim = similarities.max(axis=1).mean()
    mlflow.log_metric("test_avg_max_similarity", float(avg_sim))

    # 5) Log vectorizer to Model Registry
    mlflow.sklearn.log_model(
        sk_model=vectorizer,
        artifact_path="vectorizer",
        registered_model_name="PantryPaletteVectorizer"
    )

    # 6) Log KNN model to Model Registry
    mlflow.sklearn.log_model(
        sk_model=nn_model,
        artifact_path="nn_model",
        registered_model_name="PantryPaletteNNModel"
    )

    # 7) Save training dataset as artifact
    os.makedirs("processed_dataset", exist_ok=True)
    train_path = "processed_dataset/train_data.csv"
    train_df.to_csv(train_path, index=False)
    mlflow.log_artifact(train_path, artifact_path="training_data")

    print("TF-IDF + NearestNeighbors logged with evaluation and registered in MLflow.")