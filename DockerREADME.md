# 🍽️ PantryPalette: Smart Recipe Recommendation System

PantryPalette is a Streamlit-based web app that recommends recipes based on your available ingredients. It also logs all predictions and model training using MLflow. This setup uses Docker to deploy both the MLflow backend and the Streamlit frontend.

---

## 📦 Components

- **Streamlit App** — User interface for entering ingredients and viewing recommendations.
- **MLflow Tracking Server** — Manages experiment logging and model registry.
- **Docker Containers** — Easy deployment and isolation of services.

---

## 🚀 How to Run (Docker Setup)

### 🧰 Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))

---

### 📥 Step 1: Pull Docker Images

```bash
docker pull sandhyakilari/pantrypalette:latest
docker pull sandhyakilari/pantrypalette-mlflow:latest
```

---

### 🧠 Step 2: Run MLflow Tracking Server

```bash
docker run -d \
  --name mlflow-server \
  -p 5001:5000 \
  sandhyakilari/pantrypalette-mlflow:latest
```

- MLflow UI: http://localhost:5001

---

### 🖥️ Step 3: Run Streamlit App

```bash
docker run -d \
  --name pantrypalette-app \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 \
  -p 8501:8501 \
  sandhyakilari/pantrypalette:latest
```

- App UI: http://localhost:8501

> 💡 On Linux, if `host.docker.internal` doesn't work, replace it with the output of `ip route | grep docker0` (usually `172.17.0.1`).

---

## 🛠️ Development Mode (Optional)

To develop locally with Docker Compose:

```bash
git clone https://github.com/sandhyakilari/pantrypalette.git
cd pantrypalette
docker-compose up --build
```

Then:
- Visit [localhost:8501](http://localhost:8501) for the app
- Visit [localhost:5001](http://localhost:5001) for MLflow

---

## 🧪 Features

- Ingredient-based recipe matching
- Alternative ingredient suggestions
- MLflow experiment logging
- Cosine similarity + TF-IDF model
- Grouped views by recipe source
- Usage analytics & monitoring

---

## 🛑 Stop & Clean Containers

```bash
docker stop pantrypalette-app mlflow-server
docker rm pantrypalette-app mlflow-server
```

---

## 👩‍💻 Author

Worked By: Sandhya Kilari & Madhurya Shankar