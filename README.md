# 🍽️ PantryPalette: Your Ingredient-Inspired Recipe Guide

## 📌 Problem Statement
Many households struggle with unused ingredients, leading to food waste and repetitive meals. Finding new recipes can be time-consuming—especially for individuals with dietary restrictions.

**PantryPalette** solves this by offering a seamless, ingredient-based recipe discovery experience that helps users:
- Maximize grocery utility  
- Reduce food waste  
- Explore new and exciting meal ideas  

---

## 🚀 Features

- 🔍 **Ingredient-Based Recipe Recommendations**  
  Enter what you have and get personalized recipe suggestions.

- 🌱 **Reduce Food Waste**  
  Discover creative ways to use leftovers and pantry staples.

- 🍽️ **Minimal Add-on Ingredients**  
  Get recipes that require very few additional items.

- 🌐 **Real-Time Recipe Integration**  
  Combines dynamic web scraping with the Spoonacular API for updated results.

- 🖥️ **Interactive Interface**  
  Built with Streamlit for ease of use and accessibility.

---

## 🔄 End-to-End MLOps Pipeline

### 1. 📥 Data Ingestion
- Static dataset: [**RecipeNLG.csv**](https://huggingface.co/datasets/SandhyaKilari/RecipeNLG_dataset/resolve/main/RecipeNLG_dataset.csv)  
- Dynamic web scraping from [**Pinch of Yum Recipes**](https://github.com/sandhyakilari/pantrypalette-app/blob/main/PinchofYum_WebScraped/recipes.csv)

### 2. 🗄️ Data Storage
- Combined, cleaned, and preprocessed data stored in a [**SQLite database**](https://huggingface.co/datasets/SandhyaKilari/recipes_data/blob/main/database/recipe_data.db)

### 3. 📊 Feature Engineering & Model Training
- **TF-IDF vectorization** (1–2 grams)  
- **Cosine Similarity** using Nearest Neighbors to generate recipe matches

### 4. 📦 Model Tracking & Registry
- Tracked using **MLflow**, with models registered for dev and prod environments

### 5. 🌐 Streamlit App Deployment
A user-friendly Streamlit app delivers real-time recommendations using the trained model.

**Key Features:**
- Live recipe matching from pantry inputs  
- Smart substitutions for partial matches  
- Visual recipe cards with ingredients and instructions  
- Built-in analytics:
  - Total user queries  
  - Average response time  
  - Popular recipes  
  - Cosine similarity-based match scores  

Logs are saved for monitoring and can be visualized.

### 6. 🐳 Dockerized Application
- Containerized using **Docker** for reproducible, portable deployment

### 7. ☁️ AWS EC2 Integration
- Deployed both Streamlit and MLflow services on an **AWS EC2 (Ubuntu)** instance  
- Used **Docker Compose** to orchestrate multi-container setup  
- Ports 8501 (Streamlit) and 5000 (MLflow) exposed securely  
- Auto-restarts and logging handled via Docker for high availability  

---

## 👥 Team Members  
👩‍💻 **Madhurya Shankar**  
👩‍💻 **Sandhya Kilari**
