# 🍽️ PantryPalette: Your Ingredient-Inspired Recipe Guide

## 📌 Problem Statement
Many households struggle with unused ingredients, leading to food waste and repetitive meals. Finding new recipes can be time-consuming—especially for individuals with dietary restrictions.

**PantryPalette** addresses this by offering a seamless, ingredient-based recipe discovery experience to help users:
- Maximize grocery utility
- Reduce food waste
- Explore new and exciting meal ideas

## 🚀 Features

- 🔍 **Ingredient-Based Recipe Recommendations**  
  Enter what you have, and get personalized recipe suggestions.

- 🌱 **Reduce Food Waste**  
  Discover creative ways to use leftovers and pantry staples.

- 🍽️ **Minimal Add-on Ingredients**  
  Get recipes that require very few extra items.

- 🌐 **Real-Time Recipe Integration**  
  Pulls dynamic recipe content via web scraping and Spoonacular API.

- 🖥️ **Interactive Interface**  
  Built with Streamlit for ease of use and accessibility.

## ⚙️ Tech Stack

- **Python**, **Pandas**, **Scikit-learn**
- **SQLite** (Data Repository)
- **TF-IDF + Nearest Neighbors** (Model)
- **MLflow** (Experiment Tracking & Registry)
- **Streamlit** (App UI)
- **Docker** (Deployment)
- **Power BI** (Model Monitoring)

## 🔄 End-to-End MLOps Pipeline

### 1. 📥 Data Ingestion
- `RecipeNLG.csv` (static recipe dataset)
- Web scraped data from **PinchOfYum**

### 2. 🗄️ Data Storage
- All cleaned and processed data stored in a **SQLite** database

### 3. 📊 Feature Engineering + Model Training
- **TF-IDF vectorization** (1–2 grams)
- **Cosine Similarity Nearest Neighbors model** to recommend recipes

### 4. 📦 Model Tracking & Registry
- Use **MLflow** to track experiments and store the best model

### 5. 🌐 Streamlit App Deployment
- Integrate model predictions into a user-facing **Streamlit** application

### 6. 🐳 Dockerized Application
- Package the app using **Docker** for reproducible, portable deployment

### 7. 📈 Model Monitoring
- **Power BI** dashboard tracks:
  - Usage metrics (number of queries)
  - Average response time
  - Recipe selection trends
  - Matching effectiveness

## 👥 Team Members  
👩‍💻 **Madhurya Shankar**  
👩‍💻 **Sandhya Kilari**
