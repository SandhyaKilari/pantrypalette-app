# === Standard Library ===
import datetime
import os
import time
from collections import Counter

# === Third-Party Libraries ===
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# === MLflow Integration ===
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# === MLflow Setup ===
mlflow.set_tracking_uri("http://mlflow:5001")
client = MlflowClient()

# === Load production-aliased model versions ===
try:
    vec_alias_version = client.get_model_version_by_alias("prod.PantryPaletteVectorizer", "production").version
    model_alias_version = client.get_model_version_by_alias("prod.PantryPaletteNNModel", "production").version
except Exception as e:
    st.error(f"Failed to retrieve model versions by alias: {e}")
    st.stop()

# === Load models from registry ===
try:
    vectorizer = mlflow.sklearn.load_model(f"models:/prod.PantryPaletteVectorizer/{vec_alias_version}")
    model = mlflow.sklearn.load_model(f"models:/prod.PantryPaletteNNModel/{model_alias_version}")
except Exception as e:
    st.error(f"Failed to load models from MLflow: {e}")
    st.stop()
    
# === Load recipe metadata ===
conn = sqlite3.connect("database/recipe_data.db")
df = pd.read_sql('SELECT Title, Ingredients, Instructions, "Estimated Time", Source FROM recipes', conn)
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

# --- Initialize session state for saved favorites ----
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# --- Page Switcher ---
page = st.selectbox("Choose Page", ["🔍 Recommend Recipes", "📈 Monitor Usage", "⭐ My Favorites"])

if page == "🔍 Recommend Recipes":
    st.markdown("Enter ingredients below to get recipe suggestions and smart alternatives!")
    user_ingredients = st.text_input("🧂 Ingredients (comma-separated)", "chicken, onion, garlic")
    max_time = st.slider("⏰ Max Cooking Time (minutes)", min_value=10, max_value=240, value=60, step=10)

    if st.button("🔍 Find Recipes"):
        if user_ingredients.strip():
            start_time = time.time()
            results, indices = recommend_recipes(user_ingredients)
            duration = round(time.time() - start_time, 2)
            log_query(username, user_ingredients, duration)

            results = results[results["Estimated Time"].apply(parse_minutes) <= max_time]

            if results.empty:
                st.warning(f"😕 No recipes found within {max_time} minutes for the given ingredients.")
            else:
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

                        if st.button(f"⭐ Save '{row['Title']}' to Favorites", key=f"fav_{idx}"):
                            st.session_state.favorites.append(dict(row))
                            st.success("Added to favorites!")
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

elif page == "📈 Monitor Usage":
    log_file = "logs/user_queries.csv"
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])

        st.subheader("📊 Query Monitoring Dashboard")
        st.metric("Total Queries", len(df_log))
        st.metric("Avg Query Time (sec)", round(df_log["duration_sec"].mean(), 2))

        st.markdown("### 👤 Top Users")
        st.dataframe(df_log["user"].value_counts().head(5).reset_index().rename(columns={"index": "User", "user": "Queries"}))

        st.markdown("### 🧂 Most Common Ingredients")
        all_ings = ";".join(df_log["ingredient_list"].dropna()).split(";")
        top_ings = Counter(all_ings).most_common(10)
        st.dataframe(pd.DataFrame(top_ings, columns=["Ingredient", "Count"]))

        st.markdown("### 📅 Query Volume Over Time")
        query_by_day = df_log["timestamp"].dt.date.value_counts().sort_index()
        st.line_chart(query_by_day)

        st.markdown("### ☁️ Ingredient Word Cloud")
        wc = WordCloud(width=800, height=300, background_color='white').generate(" ".join(all_ings))
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("ℹ️ No user query logs found yet.")

elif page == "⭐ My Favorites":
    st.subheader("⭐ Saved Recipes")
    if not st.session_state.favorites:
        st.info("You haven't saved any recipes yet.")
    else:
        for idx, row in enumerate(st.session_state.favorites):
            st.markdown(f"### 🍲 {row['Title']}")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**🧂 Ingredients Needed:**")
                st.markdown(f"`{row['Ingredients']}`")
            with col2:
                st.markdown(f"**⏱️ Estimated Time:** `{row['Estimated Time']}`")
                st.markdown(f"**🌐 Source:** `{row['Source']}`")
            with st.expander("📖 Instructions"):
                st.markdown(row["Instructions"])
            st.markdown("---")