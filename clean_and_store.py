# 📥 Data Ingestion → 🧼 Cleaning → 🧪 Combining → 🧹 Instruction Preprocessing → 🧹 Step 6: Preprocess Ingredients → 🗄️ Storing into SQLite

# Import Required Libraries
import os, re, ast, sqlite3, requests, logging
import pandas as pd
from datetime import datetime

# NLTK setup
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# ===============================
# 📥 Step 1: Data Ingestion
# ===============================
def download_if_missing(path, url):
    if not os.path.exists(path):
        logging.info(f"Downloading {path}...")
        try:
            r = requests.get(url, stream=True)
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logging.info(f"Successfully downloaded: {path}")
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")
            raise

os.makedirs("dataset", exist_ok=True)

download_if_missing(
    "dataset/RecipeNLG_dataset.csv",
    "https://huggingface.co/datasets/SandhyaKilari/RecipeNLG_dataset/resolve/main/RecipeNLG_dataset.csv"
)

download_if_missing(
    "dataset/recipes.csv",
    "https://raw.githubusercontent.com/nivesayee/recipe-genie/refs/heads/main/data/raw/recipes.csv"
)

# Load data
df_static = pd.read_csv("dataset/RecipeNLG_dataset.csv")
df_static.drop(columns=["Unnamed: 0", "link", "source", "NER"], errors="ignore", inplace=True)
df_static.rename(columns={'title': 'Title', 'ingredients': 'Ingredients', 'directions': 'Instructions'}, inplace=True)
df_static['Source'] = 'RecipeNLG'

df_dynamic = pd.read_csv("dataset/recipes.csv")
df_dynamic.drop(columns=["image", "description"], errors="ignore", inplace=True)
df_dynamic.rename(columns={
    'title': 'Title',
    'ingredients': 'Ingredients',
    'instructions': 'Instructions',
    'total time': 'Estimated Time'
}, inplace=True)
df_dynamic['Source'] = 'PinchOfYum'

# ===============================
# 🧼 Step 2: Clean & Standardize
# ===============================
def extract_time_from_text(text):
    if not isinstance(text, str):
        return "Not available"
    matches = re.findall(r'(\d+)\s*(minutes|min|minute|hours|hrs|hr)', text.lower())
    total_minutes = sum(int(v) * 60 if 'hour' in u else int(v) for v, u in matches)
    return f"{total_minutes} minutes" if total_minutes else "Not available"

def cap_extreme_times(val):
    try:
        minutes = int(val.split()[0])
        return f"{minutes} minutes" if minutes <= 240 else "Not available"
    except:
        return val

logging.info("Extracting and capping estimated times...")
df_static["Estimated Time"] = df_static["Instructions"].apply(extract_time_from_text)
df_static["Estimated Time"] = df_static["Estimated Time"].apply(cap_extreme_times)

# Drop invalid rows
df_static = df_static[df_static["Estimated Time"] != "Not available"].dropna()
df_dynamic = df_dynamic.dropna().drop_duplicates()

# ================================
# 🧪 Step 3: Combine Both Sources
# ================================
receipe_df = pd.concat([df_static, df_dynamic], ignore_index=True)
receipe_df.dropna(subset=["Title", "Ingredients", "Instructions"], inplace=True)
receipe_df.drop_duplicates(subset=["Title", "Ingredients"], inplace=True)

# ================================
# 🧹 Step 4: Preprocess Instructions
# ================================
def clean_instructions(instruction_list):
    try:
        if isinstance(instruction_list, str) and instruction_list.startswith("["):
            instruction_list = ast.literal_eval(instruction_list)

        if not isinstance(instruction_list, list):
            return ""

        # Join and clean symbols
        full_text = " ".join(instruction_list)
        full_text = full_text.replace('\u00b0', '°')
        full_text = re.sub(r'(\d+)\s*°', r'\1°F', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        # Capitalize each sentence
        sentences = sent_tokenize(full_text)
        capitalized = " ".join(s.capitalize() for s in sentences)

        return capitalized

    except Exception as e:
        logging.warning(f"Instruction cleaning error: {e}")
        return ""

receipe_df["Instructions"] = receipe_df["Instructions"].apply(clean_instructions)

# ================================
# 🧹 Step 6: Preprocess Ingredients
# ================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define custom stopwords 
custom_stopwords = stop_words.union({
    # Measurement units
    'c', 'cup', 'cups', 'tsp', 'teaspoon', 'teaspoons', 'tbsp', 'tablespoon', 'tablespoons',
    'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds', 'g', 'gram', 'grams', 'kg',
    'ml', 'milliliter', 'milliliters', 'l', 'liter', 'liters', 'qt', 'quart', 'quarts',
    'pt', 'pint', 'pints', 'gal', 'gallon', 'gallons', 'pkg', 'pkgs', 'package', 'packages',
    'stick', 'sticks', 'dash', 'pinch', 'can', 'cans', 'fluid', 'fl', 'jar', 'jars',
    'box', 'boxes', 'bottle', 'bottles', 't', 'tbs', 'tbls', 'qt.', 'pt.', 'oz.', 'lb.', 'g.', 'ml.', 'kg.', 'l.', 'pkg.', 'pkt',

    # Preparation and cooking descriptors
    'chopped', 'minced', 'diced', 'sliced', 'grated', 'crushed', 'shredded', 'cut',
    'peeled', 'optional', 'seeded', 'halved', 'coarsely', 'finely', 'thinly', 'roughly',
    'cubed', 'crumbled', 'ground', 'trimmed', 'boneless', 'skinless', 'melted', 'softened',
    'cooled', 'boiled', 'cooked', 'uncooked', 'raw', 'drained', 'rinsed', 'beaten', 'size'

    # Quantity and portion descriptors
    'small', 'medium', 'large', 'extra', 'light', 'dark', 'best', 'fresh', 'freshly',
    'ripe', 'mini', 'whole', 'big', 'room', 'temperature', 'zero', 'one', 'two', 'three',
    'four', 'five', 'six', 'eight', 'ten', 'twelve', 'half', 'third', 'quarter', 'dozen',
    'thousand', 'bite'

    # Filler or generic stopwords
    'plus', 'with', 'without', 'into', 'about', 'of', 'the', 'to', 'for', 'in', 'from',
    'as', 'and', 'or', 'on', 'your', 'if', 'such', 'you', 'use', 'may', 'lightli', 'chop', 'piec', 'cube', 'grate', 'slice', 'inch', 'larg',
    'mediums', 'small', 'medium', 'best', 'big', 'light','littl', 'mediumlarg', 'mini', 'thick', 'bit-sized'
})

def preprocess_ingredients(ingredients):
    try:
        if isinstance(ingredients, str):
            ingredients_list = ast.literal_eval(ingredients) if ingredients.startswith("[") else [ingredients]
        elif isinstance(ingredients, list):
            ingredients_list = ingredients
        else:
            return ""

        cleaned_ingredients = set()

        for ing in ingredients_list:
            ing = re.sub(r'\(.*?\)', '', str(ing)).lower()  # Remove text inside parentheses and lowercase
            ing = re.sub(r'[^a-z\s]', '', ing)              # Remove all non-alphabetic characters
            tokens = word_tokenize(ing)                     # Tokenize into words
            tokens = [lemmatizer.lemmatize(token) for token in tokens 
                      if token not in custom_stopwords and len(token) > 1]  # Remove stopwords & lemmatize

            if tokens:
                phrase = " ".join(tokens)
                if "oil" not in phrase and "salt" not in phrase and "water" not in phrase:
                    cleaned_ingredients.add(phrase)

        # Return capitalized, comma-separated cleaned ingredients
        return ", ".join(sorted(ing.title() for ing in cleaned_ingredients))

    except Exception as e:
        logging.warning(f"Preprocessing error: {e}")
        return ""

receipe_df['Ingredients'] = receipe_df['Ingredients'].apply(preprocess_ingredients)

# ===============================
# 💾 Step 5.1: Save to CSV (optional)
# ===============================
os.makedirs("processed", exist_ok=True)
csv_path = os.path.join("processed", "cleaned_recipes.csv")
receipe_df.to_csv(csv_path, index=False)
logging.info(f"Saved cleaned dataset to CSV at '{csv_path}'")

# ===============================
# 🗄️ Step 5.2: Store in SQLite
# ===============================
os.makedirs("database", exist_ok=True)
conn = sqlite3.connect("database/recipe_data.db")
receipe_df.to_sql("recipes", conn, if_exists="replace", index=False)
conn.close()
logging.info(f"Stored {len(receipe_df)} recipes in 'database/recipe_data.db'")
