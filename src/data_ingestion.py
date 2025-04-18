# 📥 Data Ingestion → 🧼 Cleaning → 🧪 Combining → 🧹 Instruction Preprocessing → 🧹 Step 6: Preprocess Ingredients → 🗄️ Storing into SQLite

# ===========================
# 📥 Step 1: Data Ingestion
# ===========================
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import os
import pandas as pd
import ast
import re
from ast import literal_eval

import sqlite3

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Load RecipeNLG Dataset
df_static = pd.read_csv("/Users/sandhyakilari/Desktop/pantrypalette-app/dataset/RecipeNLG_dataset.csv")
df_static.drop(columns=["Unnamed: 0", "link", "source", "NER"], inplace=True)
df_static.rename(columns={'title': 'Title', 'ingredients': 'Ingredients', 'directions': 'Instructions'}, inplace=True)
df_static['Source'] = 'RecipeNLG'

# Load PinchOfYum Web Scraped Dataset
df_dynamic = pd.read_csv("/Users/sandhyakilari/Desktop/pantrypalette-app/dataset/recipes.csv")
df_dynamic.drop(columns=["image", "description"], inplace=True)
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
    total_minutes = 0
    for value, unit in matches:
        value = int(value)
        if 'hour' in unit or 'hr' in unit:
            total_minutes += value * 60
        else:
            total_minutes += value
    return f"{total_minutes} minutes" if total_minutes > 0 else "Not available"

def cap_extreme_times(val):
    try:
        minutes = int(val.split()[0])
        return f"{minutes} minutes" if minutes <= 240 else "Not available"
    except:
        return val

df_static["Estimated Time"] = df_static["Instructions"].apply(extract_time_from_text)
df_static["Estimated Time"] = df_static["Estimated Time"].apply(cap_extreme_times)
df_static = df_static[df_static["Estimated Time"] != "Not available"].dropna()

df_dynamic = df_dynamic.dropna().drop_duplicates()

# ================================
# 🧪 Step 3: Combine Both Sources
# ================================
receipe_df = pd.concat([df_static, df_dynamic], ignore_index=True)
receipe_df.dropna()

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
        print(f"Instruction cleaning error: {e}")
        return ""

receipe_df["Instructions"] = receipe_df["Instructions"].apply(clean_instructions)

# ================================
# 🧹 Step 6: Preprocess Ingredients
# ================================
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

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
        print(f"Preprocessing error: {e}")
        return ""

receipe_df['Ingredients'] = receipe_df['Ingredients'].apply(preprocess_ingredients)

# ===============================
# 🗄️ Step 5: Store in SQLite
# ===============================
os.makedirs("../database", exist_ok=True)

conn = sqlite3.connect("../database/recipe_data.db")
receipe_df.to_sql("recipes", conn, if_exists="replace", index=False)
conn.close()

print("Combined and cleaned dataset stored in 'recipe_data.db' under 'recipes' table.")