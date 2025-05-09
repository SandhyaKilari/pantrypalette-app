# 🍽️ PantryPalette: Smart Recipe Recommendation System

PantryPalette is a Streamlit-based web app that recommends recipes based on your available ingredients. It logs all user interactions and model training metadata via MLflow. Both services are containerized and deployed using Docker.

---

## 📦 Components

- **Streamlit App** — User interface to enter ingredients and view recipe recommendations.
- **MLflow Server** — Tracks model experiments, logs, and registry.
- **Docker Compose** — Used to deploy both services on any host (e.g., EC2).

---

## 🚀 Deploy on AWS EC2 (Ubuntu)

### 🧰 Prerequisites

- An AWS EC2 instance (Ubuntu 20.04 or later)
- Security group open to:
  - Port `22` for SSH
  - Port `8501` for Streamlit
  - Port `5000` for MLflow
- Docker & Docker Compose installed:
  ```bash
  sudo apt update
  sudo apt install docker.io docker-compose -y
  sudo usermod -aG docker $USER
  newgrp docker
  ```

---

### 📥 Step 1: Pull Docker Images

```bash
docker pull sandhyakilari/pantrypalette:latest
docker pull sandhyakilari/pantrypalette-mlflow:latest
```

---

### 📂 Step 2: Upload Project Files

Upload the following files to your EC2 instance:

- `docker-compose.yaml`
- Create empty folders:
  ```bash
  mkdir logs database mlruns
  ```

You can use `scp` from your local machine:

```bash
scp docker-compose.yaml ubuntu@<EC2-IP>:/home/ubuntu/
```

Or use `Git` as it's hosted on GitHub:

```bash
git clone https://github.com/sandhyakilari/pantrypalette.git
cd pantrypalette
```

---

### 🧠 Step 3: Launch Services with Docker Compose

```bash
docker-compose up -d
```

This will:
- Start MLflow at `http://<EC2-IP>:5000`
- Start Streamlit at `http://<EC2-IP>:8501`

---

### 🔗 Sample Output

- **Streamlit UI** → http://`<your-ec2-ip>`:8501
- **MLflow UI** → http://`<your-ec2-ip>`:5000

You can access these URLs in your browser (make sure the security group allows inbound HTTP access to these ports).

---

## 🧪 Key Features

- TF-IDF + Cosine Similarity for ingredient matching
- Grouped views by recipe source
- Alternative ingredient suggestions
- MLflow experiment logging and versioning
- Local logging of queries for usage analytics
- Dockerized deployment for easy reproducibility

---

## 🛑 Stop & Clean

To stop and remove containers:

```bash
docker-compose down
```

To remove volumes:

```bash
docker-compose down -v
```

---

## 👩‍💻 Authors

Created by:
- **Sandhya Kilari**  
- **Madhurya Shankar**
