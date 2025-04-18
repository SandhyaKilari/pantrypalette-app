
import streamlit as st
import pandas as pd
import mlflow.pyfunc
import mlflow.sklearn
import sqlite3
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load SQLite recipe data
@st.cache_data
def load_recipes():
    conn = sqlite3.connect("../database/recipe_data.db")
    df = pd.read_sql_query("SELECT Title, Ingredients FROM recipes", conn)
    conn.close()
    return df

# Load model and vectorizer from MLflow Model Registry
@st.cache_resource
def load_model_and_vectorizer():
    vectorizer = mlflow.sklearn.load_model("models:/PantryPaletteVectorizer/Production")
    nn_model = mlflow.sklearn.load_model("models:/PantryPaletteNNModel/Production")
    return vectorizer, nn_model

# Get top similar recipes
def recommend_recipes(user_input, vectorizer, model, recipes_df):
    input_vec = vectorizer.transform([user_input])
    similarities, indices = model.kneighbors(input_vec, n_neighbors=10)
    return recipes_df.iloc[indices[0]].copy(), similarities[0]

# Streamlit UI
st.set_page_config(page_title="PantryPalette - Recipe Recommender", layout="wide")
st.title("🍽️ PantryPalette: Ingredient-Based Recipe Recommender")

# Load data and models
recipes_df = load_recipes()
vectorizer, nn_model = load_model_and_vectorizer()

# Input
user_ingredients = st.text_area("Enter ingredients (comma-separated):", height=100)

if st.button("🔍 Recommend Recipes") and user_ingredients.strip():
    with st.spinner("Finding recipes..."):
        results, scores = recommend_recipes(user_ingredients, vectorizer, nn_model, recipes_df)
        results["Similarity Score"] = (1 - scores).round(2)

        st.subheader("🔗 Top Recipe Matches")
        st.dataframe(results[["Title", "Ingredients", "Similarity Score"]].reset_index(drop=True))
