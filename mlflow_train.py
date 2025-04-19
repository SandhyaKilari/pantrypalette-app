import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import logging  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from mlflow.tracking import MlflowClient

# === Setup logging ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/train_model.log"),
        logging.StreamHandler()
    ]
)

# === Load data from SQLite database ===
db_path = os.path.join("database", "recipe_data.db")
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT Title, Ingredients FROM recipes", conn)
conn.close()

# Remove rows with missing or empty ingredients
df = df[df["Ingredients"].notnull() & (df["Ingredients"] != "")]
logging.info(f"Loaded {len(df)} recipes from database.")

# === Optional: Use smaller sample for dev speed ===
df = df.sample(n=30000, random_state=42)

# === Train-test split (80/20) ===
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# === Define hyperparameter grid for tuning ===
param_grid = {
    "tfidf__max_features": [500, 1000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "nn__n_neighbors": [5, 10],
    "nn__metric": ["cosine"],
}

# Generate all combinations
search_space = [
    (mf, ngr, nn, mt)
    for mf in param_grid["tfidf__max_features"]
    for ngr in param_grid["tfidf__ngram_range"]
    for nn in param_grid["nn__n_neighbors"]
    for mt in param_grid["nn__metric"]
]

# === Set up MLflow experiment ===
mlflow.set_tracking_uri("http://localhost:5001") # Local Tracking Server
logging.info(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
mlflow.set_experiment("PantryPalette_Recipe_Recommendation")

# Track best model
best_score = -np.inf
best_vectorizer, best_model, best_params = None, None, None

# === Grid search through parameter combinations ===
for max_features, ngram_range, n_neighbors, metric in search_space:
    run_name = f"TFIDF_{max_features}_NGRAM_{ngram_range}_K_{n_neighbors}_METRIC_{metric}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "stage": "training",
            "developer": "Sandhya Kilari and Madhurya Shankar",
            "model_type": "TFIDF + NearestNeighbors"
        })

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
        X_train = vectorizer.fit_transform(train_df["Ingredients"])
        X_test = vectorizer.transform(test_df["Ingredients"])

        # 2. Nearest Neighbors model
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn_model.fit(X_train)

        # 3. Evaluation: efficient cosine similarity (1 - distance)
        distances, _ = nn_model.kneighbors(X_test)
        avg_sim = float(1 - distances.min(axis=1).mean())

        # 4. Log parameters and metric to MLflow
        mlflow.log_params({
            "tfidf_max_features": max_features,
            "tfidf_ngram_range": ngram_range,
            "nn_n_neighbors": n_neighbors,
            "nn_metric": metric
        })
        
        mlflow.log_metric("test_avg_max_cosine_similarity", avg_sim)
        logging.info(f"[{run_name}] Avg Max Cosine Similarity: {avg_sim:.4f}")

        # 5. Save best model artifacts
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

            # Register best model + vectorizer with MLflow
            mlflow.sklearn.log_model(vectorizer, "best_tfidf_vectorizer")
            mlflow.sklearn.log_model(nn_model, "best_nearest_neighbors_model")
            logging.info(f"New best model found! Score: {best_score:.4f}")

# === Save best model/vectorizer locally ===
os.makedirs("model", exist_ok=True)

try:
    joblib.dump(best_vectorizer, os.path.join("model", "tfidf_vectorizer.pkl"))
    joblib.dump(best_model, os.path.join("model", "nearest_neighbors.pkl"))
    logging.info("Best model artifacts saved to 'model/' folder.")
except Exception as e:
    logging.error(f"Failed to save model artifacts: {e}")

# MLflow model registration
# === Helper Function: Assign alias to model version based on stage ===
def assign_alias_to_stage(model_name, stage, alias):
    try:
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if not latest_versions:
            raise ValueError(f"No model versions found for '{model_name}' in stage '{stage}'.")

        latest_mv = latest_versions[0]
        client.set_registered_model_alias(model_name, alias, latest_mv.version)
        logging.info(f"Alias '{alias}' assigned to version {latest_mv.version} of '{model_name}'")
    except Exception as e:
        logging.warning(f"Failed to assign alias '{alias}' for '{model_name}': {e}")

# === Register final best model to MLflow registry and assign aliases ===
client = MlflowClient()

# === Register models in 'dev' environment ===
mlflow.sklearn.log_model(
    sk_model=best_vectorizer,
    artifact_path="vectorizer",
    registered_model_name="dev.PantryPaletteVectorizer"
)
mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    registered_model_name="dev.PantryPaletteNNModel"
)

# Get latest versions just logged
vec_version = client.get_latest_versions("dev.PantryPaletteVectorizer", stages=["None"])[0].version
model_version = client.get_latest_versions("dev.PantryPaletteNNModel", stages=["None"])[0].version

# Assign dev aliases
client.set_registered_model_alias("dev.PantryPaletteVectorizer", "dev", vec_version)
client.set_registered_model_alias("dev.PantryPaletteNNModel", "dev", model_version)

logging.info("Registered and aliased dev models.")

# Load from dev
vectorizer_uri = f"models:/dev.PantryPaletteVectorizer@dev"
model_uri = f"models:/dev.PantryPaletteNNModel@dev"

# Load the models
vectorizer = mlflow.sklearn.load_model(vectorizer_uri)
nn_model = mlflow.sklearn.load_model(model_uri)

# Register to prod
mlflow.sklearn.log_model(
    sk_model=vectorizer,
    artifact_path="vectorizer",
    registered_model_name="prod.PantryPaletteVectorizer"
)
mlflow.sklearn.log_model(
    sk_model=nn_model,
    artifact_path="model",
    registered_model_name="prod.PantryPaletteNNModel"
)

# Get version numbers in prod
prod_vec_version = client.get_latest_versions("prod.PantryPaletteVectorizer", stages=["None"])[0].version
prod_model_version = client.get_latest_versions("prod.PantryPaletteNNModel", stages=["None"])[0].version

# Assign production aliases
client.set_registered_model_alias("prod.PantryPaletteVectorizer", "production", prod_vec_version)
client.set_registered_model_alias("prod.PantryPaletteNNModel", "production", prod_model_version)

logging.info(f"Promoted to prod. Vectorizer v{prod_vec_version}, Model v{prod_model_version}")

logging.info(f"Best parameters: {best_params}")
logging.info(f"Best score: {best_score:.4f}")