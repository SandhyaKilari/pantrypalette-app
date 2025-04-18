import streamlit as st
import joblib
import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import datetime
from collections import Counter
import time

# === Load model artifacts ===
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = joblib.load("model/nearest_neighbors.pkl")

# === Load recipe metadata ===
conn = sqlite3.connect("database/recipe_data.db")
df = pd.read_sql("SELECT Title, Ingredients, Instructions, [Estimated Time], Source FROM recipes", conn)
conn.close()

# === Helper: Recommend recipes ===
def recommend_recipes(user_input, top_n=5):
    user_vec = vectorizer.transform([user_input])
    distances, indices = model.kneighbors(user_vec, n_neighbors=top_n)
    return df.iloc[indices[0]].copy(), indices[0]

# === Helper: Recommend alternative recipes with minimal new ingredients ===
def suggest_alternatives(input_ings, top_n=5):
    def ingredient_set(text):
        return set(map(str.strip, text.lower().split(',')))
    input_set = ingredient_set(input_ings)
    df["missing_count"] = df["Ingredients"].apply(lambda x: len(ingredient_set(x) - input_set))
    alternatives = df[df["missing_count"] > 0].sort_values("missing_count").head(top_n)
    return alternatives

# === Helper: Group by Source or Category ===
def group_by_source(recipes):
    grouped = recipes.groupby("Source").apply(lambda x: x[["Title", "Estimated Time"]].to_dict("records"))
    return grouped.to_dict()

# === Helper: Log query ===
def log_query(user, ingredients, duration):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "user_queries.csv")
    now = datetime.datetime.now().isoformat()
    ing_list = [i.strip() for i in ingredients.split(',')]
    row = pd.DataFrame([[now, user, ingredients, duration, ";".join(ing_list)]], columns=["timestamp", "user", "ingredients", "duration_sec", "ingredient_list"])
    if os.path.exists(log_path):
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row.to_csv(log_path, index=False)

# === Helper: Parse Estimated Time as minutes ===
def parse_minutes(time_str):
    try:
        return int(time_str.split()[0])
    except:
        return 9999

# === Streamlit UI ===
st.set_page_config(page_title="PantryPalette", layout="wide")
st.title("🥘 PantryPalette: Smart Recipe Recommender")

# --- Optional Login ---
st.sidebar.subheader("🔐 Login")
username = st.sidebar.text_input("Username", value="guest")

st.markdown("Enter ingredients below to get recipe suggestions and smart alternatives!")
user_ingredients = st.text_input("🧂 Ingredients (comma-separated)", "chicken, onion, garlic")

# Add slider for estimated cooking time filter
max_time = st.slider("⏰ Max Cooking Time (minutes)", min_value=10, max_value=240, value=60, step=10)

if st.button("🔍 Find Recipes"):
    if user_ingredients.strip():
        start_time = time.time()
        results, indices = recommend_recipes(user_ingredients)
        duration = round(time.time() - start_time, 2)
        log_query(username, user_ingredients, duration)

        # Filter results by max estimated time
        results = results[results["Estimated Time"].apply(parse_minutes) <= max_time]

        st.subheader(f"🍽️ Top {len(results)} Recipes (within {max_time} mins):")
        for idx, row in results.iterrows():
            with st.container():
                st.markdown(f"### 🍲 {row['Title']}")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**🧂 Ingredients Needed:**")
                    st.markdown(f"`{row['Ingredients']}`")
                with col2:
                    st.markdown(f"**⏱️ Estimated Time:** `{row['Estimated Time']}`")
                    st.markdown(f"**🌐 Source:** `{row['Source']}`")
                with st.expander("📖 Click to view instructions"):
                    st.markdown(row['Instructions'])
                st.markdown("---")

        st.subheader("🧠 Alternative Recipes with Minimal New Ingredients:")
        alternatives = suggest_alternatives(user_ingredients, top_n=5)
        for _, row in alternatives.iterrows():
            st.markdown(f"**{row['Title']}** — Missing Ingredients: {row['missing_count']} | Source: {row['Source']}")

        st.subheader("📊 Grouped Recommendations by Source")
        grouped = group_by_source(results)
        for source, entries in grouped.items():
            st.markdown(f"**{source}**:")
            for item in entries:
                st.markdown(f"- {item['Title']} ({item['Estimated Time']})")
    else:
        st.warning("Please enter at least one ingredient.")