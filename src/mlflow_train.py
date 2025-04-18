import pandas as pd
import sqlite3
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load cleaned ingredients
conn = sqlite3.connect("recipe_data.db")
df = pd.read_sql_query("SELECT Title, Ingredients FROM recipes", conn)
conn.close()

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # or your remote MLflow server
mlflow.set_experiment("PantryPalette_Recipe_Recommendation")

with mlflow.start_run(run_name="TFIDF + NearestNeighbors"):
    # Vectorize
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X = vectorizer.fit_transform(df["Ingredients"])
    
    # Train model
    model = NearestNeighbors(n_neighbors=10, metric="cosine")
    model.fit(X)
    
    # Log params
    mlflow.log_param("ngram_range", (1, 2))
    mlflow.log_param("max_features", 1000)
    mlflow.log_param("n_neighbors", 10)
    mlflow.log_param("metric", "cosine")
    
    # Log artifacts
    mlflow.sklearn.log_model(vectorizer, "tfidf_vectorizer")
    mlflow.sklearn.log_model(model, "nearest_neighbors_model")
    
    print("Model and vectorizer logged to MLflow.")