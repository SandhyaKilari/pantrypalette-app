# === PantryPalette – Professional Edition (Favorites, Monitoring & Alternatives) ===
# Implements user authentication, persistent favorites, advanced monitoring,
# configurable recommendation count, improved UI, and n-1 alternative recipes.

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import datetime
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Third‑Party Libraries
# ---------------------------------------------------------------------------
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401 (future use)
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# MLflow Integration
# ---------------------------------------------------------------------------
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# MLflow Setup
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()

# ---------------------------------------------------------------------------
# App & Database Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "database"
DATA_DIR.mkdir(exist_ok=True)
USER_DB_PATH = APP_DIR / "database" / "users.db"
RECIPE_DB_PATH = APP_DIR / "database" / "recipe_data.db"
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "user_queries.csv"

# ---------------------------------------------------------------------------
# Utility: database helpers
# ---------------------------------------------------------------------------
def get_conn(db_path: Path):
    return sqlite3.connect(db_path, check_same_thread=False)

# ---------------------------------------------------------------------------
# User Auth
# ---------------------------------------------------------------------------
def init_user_db():
    conn = get_conn(USER_DB_PATH)
    with conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                         username TEXT PRIMARY KEY,
                         password TEXT NOT NULL,
                         created_at TEXT NOT NULL
                     )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS favorites (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         username TEXT NOT NULL,
                         title TEXT,
                         ingredients TEXT,
                         instructions TEXT,
                         est_time TEXT,
                         source TEXT,
                         saved_at TEXT,
                         UNIQUE(username, title)
                     )"""
        )
    conn.close()


def register_user(username: str, password: str) -> bool:
    if not username or not password:
        return False
    conn = get_conn(USER_DB_PATH)
    try:
        with conn:
            conn.execute(
                "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                (username, password, datetime.datetime.utcnow().isoformat()),
            )
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate(username: str, password: str) -> bool:
    conn = get_conn(USER_DB_PATH)
    cur = conn.execute(
        "SELECT 1 FROM users WHERE username = ? AND password = ?",
        (username, password),
    )
    ok = cur.fetchone() is not None
    conn.close()
    return ok

# ---------------------------------------------------------------------------
# Favorites CRUD
# ---------------------------------------------------------------------------
def add_favorite(username: str, row: dict):
    conn = get_conn(USER_DB_PATH)
    with conn:
        conn.execute(
            """INSERT OR IGNORE INTO favorites
               (username, title, ingredients, instructions, est_time, source, saved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                username,
                row["Title"].strip(),
                row["Ingredients"],
                row["Instructions"],
                row["estimated_time"],
                row["Source"],
                datetime.datetime.utcnow().isoformat(),
            ),
        )
    conn.close()


def remove_favorite(username: str, title: str):
    conn = get_conn(USER_DB_PATH)
    with conn:
        conn.execute(
            "DELETE FROM favorites WHERE username = ? AND title = ?", 
            (username, title.strip()))
    conn.close()


def fetch_favorites(username: str) -> pd.DataFrame:
    conn = get_conn(USER_DB_PATH)
    fav_df = pd.read_sql(
        "SELECT title, ingredients, instructions, est_time, source, saved_at FROM favorites WHERE username = ?",
        conn,
        params=(username,),
    )
    conn.close()
    return fav_df

# Initialize user DB
init_user_db()

# ---------------------------------------------------------------------------
# Load production‑aliased model versions
# ---------------------------------------------------------------------------
try:
    vec_alias_version = client.get_model_version_by_alias(
        "prod.PantryPaletteVectorizer", "production"
    ).version
    model_alias_version = client.get_model_version_by_alias(
        "prod.PantryPaletteNNModel", "production"
    ).version
except Exception as e:
    st.error(f"Failed to retrieve model aliases: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
try:
    vectorizer = mlflow.sklearn.load_model(
        f"models:/prod.PantryPaletteVectorizer/{vec_alias_version}"
    )
    model = mlflow.sklearn.load_model(
        f"models:/prod.PantryPaletteNNModel/{model_alias_version}"
    )
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Load recipe metadata
# ---------------------------------------------------------------------------
conn = get_conn(RECIPE_DB_PATH)
df_recipes = pd.read_sql(
    '''SELECT Title, Ingredients, Instructions, 
              "Estimated Time" AS estimated_time, 
              Source 
       FROM recipes''',
    conn
)
conn.close()

# ---------------------------------------------------------------------------
# Helpers for recommendation logic
# ---------------------------------------------------------------------------
def parse_minutes(time_str: str) -> int:
    try:
        return int(str(time_str).split()[0])
    except:
        return 9999


def filter_by_user_ingredients(user_ings: str) -> pd.Series:
    user_set = {ing.strip().lower() for ing in user_ings.split(',') if ing.strip()}
    return df_recipes['Ingredients'].apply(
        lambda text: user_set.issubset({i.strip().lower() for i in text.split(',')})
    )


def highlight_ingredients(ingredients_str: str, user_input: str, color: str = "#4CAF50") -> str:
    """Highlight user-provided ingredients in recipe ingredient lists."""
    user_ingredients = {ing.strip().lower() for ing in user_input.split(',')}
    ingredients = [ing.strip() for ing in ingredients_str.split(',')]
    highlighted = []
    for ing in ingredients:
        if ing.lower() in user_ingredients:
            highlighted.append(f"<span style='color: {color}; font-weight: bold;'>{ing}</span>")
        else:
            highlighted.append(ing)
    return ', '.join(highlighted)


def recommend_recipes(user_input: str, top_n: int) -> pd.DataFrame:
    mask = filter_by_user_ingredients(user_input)
    if not mask.any():
        return pd.DataFrame()
    user_vec = vectorizer.transform([user_input])
    dists, idxs = model.kneighbors(user_vec, n_neighbors=model.n_samples_fit_)
    dists, idxs = dists.flatten(), idxs.flatten()
    filtered = [(i, d) for i, d in zip(idxs, dists) if mask.iat[i]]
    sel_idxs = [i for i, _ in filtered[:top_n]]
    return df_recipes.iloc[sel_idxs].copy()


def suggest_alternatives(input_ings: str, top_n: int = 5, max_time: int = None) -> pd.DataFrame:
    user_list = [i.strip().lower() for i in input_ings.split(",") if i.strip()]
    N = len(user_list)
    user_set = set(user_list)
    temp = df_recipes.copy()
    temp['ing_set'] = temp['Ingredients'].apply(
        lambda x: {i.strip().lower() for i in x.split(',')}
    )
    temp['common_count'] = temp['ing_set'].apply(lambda s: len(s & user_set))
    alts = temp[(temp['common_count'] >= N - 1) & (temp['common_count'] < N)].copy()
    if max_time is not None:
        alts['time_min'] = alts['estimated_time'].apply(parse_minutes)
        alts = alts[alts['time_min'] <= max_time]
    alts['missing_count'] = N - alts['common_count']
    alts['missing_ingredients'] = alts['ing_set'].apply(lambda s: user_set - s)
    return alts.sort_values(['missing_count', 'time_min']).head(top_n)


def group_by_source(recipes: pd.DataFrame) -> dict:
    grouped = recipes.groupby('Source').apply(
        lambda x: x[['Title','estimated_time']].to_dict('records')
    )
    return grouped.to_dict()


def log_query(user: str, ingredients: str, duration: float):
    now = datetime.datetime.utcnow().isoformat()
    ing_list = [i.strip() for i in ingredients.split(',')]
    row = pd.DataFrame(
        [[now, user, ingredients, duration, ';'.join(ing_list)]],
        columns=['timestamp','user','ingredients','duration_sec','ingredient_list']
    )
    if LOG_PATH.exists():
        row.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        row.to_csv(LOG_PATH, index=False)

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title='PantryPalette', layout='wide', page_icon='🍳')

# Sidebar – Auth
st.sidebar.title('PantryPalette')
if 'auth_page' not in st.session_state:
    st.session_state.auth_page = 'Login'

# Sidebar - Auth
auth_choice = st.sidebar.radio(
    'Account',
    ['Login', 'Register'],
    index=0 if st.session_state.auth_page == 'Login' else 1,
    key='auth_radio'
)

# Add to session state initialization
if 'reg_key_suffix' not in st.session_state:
    st.session_state.reg_key_suffix = 0

# Modified registration section
if auth_choice == 'Register':
    st.sidebar.subheader('Create Account')
    # Use dynamic keys based on suffix
    reg_user = st.sidebar.text_input(
        '👤 Username', 
        key=f'reg_user_{st.session_state.reg_key_suffix}'
    )
    reg_pass = st.sidebar.text_input(
        '🔑 Password', 
        type='password', 
        key=f'reg_pass_{st.session_state.reg_key_suffix}'
    )
    
    if st.sidebar.button('✍️ Register'):
        if register_user(reg_user, reg_pass):
            st.sidebar.success('Registration successful. Please login.')
            # Update auth state
            st.session_state.auth_page = 'Login'
            # Change key suffix to reset widgets next time
            st.session_state.reg_key_suffix += 1
        else:
            st.sidebar.error('Username exists or invalid.')

else:  # Login section
    st.sidebar.subheader('Login')
    login_user = st.sidebar.text_input(
        'Username', 
        key=f'login_user_{st.session_state.reg_key_suffix}'
    )
    login_pass = st.sidebar.text_input(
        'Password', 
        type='password', 
        key=f'login_pass_{st.session_state.reg_key_suffix}'
    )
    
    if st.sidebar.button('🔐 Login'):
        if authenticate(login_user, login_pass):
            st.session_state.user = login_user
            st.sidebar.success(f'Welcome, {login_user}!')
            st.session_state.favorites_df = fetch_favorites(login_user)
        else:
            st.sidebar.error('Invalid credentials.')

current_user = st.session_state.get('user')
fav_columns = ['title', 'ingredients', 'instructions', 'est_time', 'source', 'saved_at']

if current_user:
    st.session_state.setdefault('favorites_df', fetch_favorites(current_user))
else:
    # Initialize empty DataFrame with proper columns
    st.session_state.favorites_df = pd.DataFrame(columns=fav_columns)
    st.info('Please login via the sidebar to save recipes or access the dashboard.')

def on_save(row: dict):
    if current_user:
        add_favorite(current_user, row)
        st.session_state.favorites_df = fetch_favorites(current_user)
        st.session_state.saved_title = row['Title']
        st.rerun()
    else:
        st.session_state.save_login_required = True

def on_remove(title: str):
    if current_user:
        remove_favorite(current_user, title)
        st.session_state.favorites_df = fetch_favorites(current_user)
        st.rerun()

# Navigation
page = st.selectbox('Navigate', ['🔍 Recommend Recipes', '📊 Dashboard & Monitoring', '⭐ My Favorites'])

if page == '🔍 Recommend Recipes':
    st.title('🥘 Recipe Recommendations')
    user_ings = st.text_input('🧂 Ingredients (comma‑separated)', key='user_input')
    max_time = st.slider('⏰ Max Cook Time (mins)', 10, 240, 60, 10, key='max_time')
    top_n = st.slider('📑 Number of Recipies', 1, 10, 6, key='top_n')

    if 'saved_title' in st.session_state:
        st.success(f"Saved '{st.session_state.saved_title}' to favorites!")
        del st.session_state.saved_title

    if st.button('🔍 Find Recipes'):
        if not user_ings.strip():
            st.warning('Enter at least one ingredient.')
        else:
            t0 = time.time()
            results = recommend_recipes(user_ings, top_n)
            duration = round(time.time() - t0, 2)
            if current_user:
                log_query(current_user, user_ings, duration)
            results = results[results['estimated_time'].apply(parse_minutes) <= max_time]
            st.session_state['latest_results'] = results
            if results.empty:
                st.warning(f'No recipes ≤ {max_time} mins containing all: {user_ings}')
            else:
                st.success(f'Showing {len(results)} recipes in {duration}s')
                for i in range(0, len(results), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i+j >= len(results): break
                        row = results.iloc[i+j]
                        title_norm = row['Title'].strip().lower()
                        saved_norms = st.session_state.favorites_df['title'].str.strip().str.lower()
                        saved = title_norm in saved_norms.values
                        with cols[j]:
                            st.markdown(f"#### 🍲 {row['Title']}")
                            st.markdown(f"**⏱️ {row['estimated_time']}** | 🌐 {row['Source']}")
                            highlighted_ings = highlight_ingredients(row['Ingredients'], user_ings)
                            st.markdown(f"**Ingredients:** {highlighted_ings}", unsafe_allow_html=True)
                            with st.expander('📖 Instructions'):
                                st.markdown(row['Instructions'])
                            if saved:
                                st.button('🗑️ Remove', 
                                        key=f"rem_{row['Title']}",
                                        on_click=on_remove, 
                                        args=(row['Title'],))
                            else:
                                st.button('⭐ Save', 
                                        key=f"save_{row['Title']}",
                                        on_click=on_save, 
                                        args=(row,))                                
                if st.session_state.get('save_login_required'):
                    st.warning('Please login or register to save recipes.')
                    del st.session_state.save_login_required
                st.markdown('---')
                st.subheader('🧠 Alternative Suggestions')
                alt_df = suggest_alternatives(user_ings, top_n=5, max_time=max_time)
                if alt_df.empty:
                    st.info('No alternatives found.')
                else:
                    for _, r in alt_df.iterrows():
                        missing_ing = ', '.join(r['missing_ingredients'])
                        with st.expander(f"{r['Title']}"):
                            alt_highlighted = highlight_ingredients(r['Ingredients'], user_ings)
                            st.markdown(f"**Ingredients:** {alt_highlighted}", unsafe_allow_html=True)
                            st.markdown(f"**Excluding:** <span style='color: #ff0000;'>{missing_ing}</span>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"**Instructions:** {r['Instructions']}")

elif page == '📊 Dashboard & Monitoring':
    st.title('📈 Analytics Dashboard')
    if not current_user:
        st.warning('Please login to access the dashboard.')
        st.stop()
    
    df_log = pd.read_csv(LOG_PATH, parse_dates=['timestamp']) if LOG_PATH.exists() else pd.DataFrame()
    df_fav_all = fetch_favorites(current_user)
    conn_u = get_conn(USER_DB_PATH)
    df_fav_global = pd.read_sql('SELECT * FROM favorites', conn_u)
    conn_u.close()
    if not df_log.empty:
        mind, maxd = df_log['timestamp'].min().date(), df_log['timestamp'].max().date()
        sd, ed = st.date_input('Date range', (mind, maxd), min_value=mind, max_value=maxd)
        df_log = df_log[(df_log['timestamp'].dt.date>=sd) & (df_log['timestamp'].dt.date<=ed)]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Total Queries', len(df_log))
    c2.metric('Unique Users', df_log['user'].nunique())
    c3.metric('Avg Duration (s)', round(df_log['duration_sec'].mean(),2) if not df_log.empty else 0)
    c4.metric('Total Favorites', len(df_fav_global))
    tabs = st.tabs(['📈 Queries','⭐ Favorites','🔗 Source Breakdown'])
    with tabs[0]:
        if df_log.empty:
            st.info('No data.')
        else:
            st.line_chart(df_log['timestamp'].dt.date.value_counts().sort_index())
            st.dataframe(df_log['user'].value_counts().head(10).reset_index().rename(columns={'index':'user','user':'UserName'}))
            all_ing = ";".join(df_log['ingredient_list'].dropna()).split(';')
            st.dataframe(pd.DataFrame(Counter(all_ing).most_common(20), columns=['Ingredient','Count']))
    with tabs[1]:
        if df_fav_global.empty:
            st.info('No global favorites.')
        else:
            st.dataframe(df_fav_global.groupby('title')['username'].count().sort_values(ascending=False).head(20).reset_index().rename(columns={'username':'Saves'}))
        if df_fav_all.empty:
            st.info('You have no favorites.')
        else:
            st.area_chart(pd.to_datetime(df_fav_all['saved_at']).dt.date.value_counts().sort_index())
    with tabs[2]:
        latest = st.session_state.get('latest_results', pd.DataFrame())
        if latest.empty:
            st.info('Run a search first.')
        else:
            for src, items in group_by_source(latest).items():
                with st.expander(f"{src} ({len(items)})"):
                    for itm in items:
                        st.markdown(f"• {itm['Title']} – {itm['estimated_time']}")

else:
    st.title('⭐ My Favorites')
    if not current_user:
        st.warning('Please login to view your favorites.')
        st.stop()
    
    fav_df = st.session_state.favorites_df
    if fav_df.empty:
        st.info("You haven't saved any recipes yet.")
    else:
        for _, row in fav_df.iterrows():
            with st.container():
                st.markdown(f"### 🍲 {row['title']}")
                st.markdown(f"**⏱️ {row['est_time']}** | 🌐 {row['source']}")
                c1,c2 = st.columns([2,1])
                with c1:
                    if 'user_input' in st.session_state:
                        highlighted = highlight_ingredients(row['ingredients'], st.session_state.user_input)
                        st.markdown(f"**Ingredients:** {highlighted}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Ingredients:** {row['ingredients']}")
                with c2:
                    with st.expander('📖 Instructions'):
                        st.markdown(row['instructions'])
                    st.button('🗑️ Remove', key=f"del_{row['title']}", on_click=on_remove, args=(row['title'],))
        st.download_button('⬇️ Download CSV', fav_df.to_csv(index=False), file_name='my_favorites.csv', mime='text/csv')

# Footer
st.markdown(
    """
    <hr />
    <p style='text-align:center;font-size:0.9rem;'>
      Developed by
      <a href='https://www.linkedin.com/in/madhurya-shankar-7344541b2' target='_blank'>Madhurya Shankar</a>
      &amp;
      <a href='https://www.linkedin.com/in/sandhya-kilari' target='_blank'>Sandhya Kilari</a>
      © 2025 PantryPalette
    </p>
    """,
    unsafe_allow_html=True,
)