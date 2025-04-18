# syntax=docker/dockerfile:1

# === Base image ===
FROM python:3.10-slim

# === Set working directory ===
WORKDIR /app

# === Install system dependencies ===
RUN apt-get update && \
    apt-get install -y build-essential gcc curl && \
    rm -rf /var/lib/apt/lists/*

# === Copy project files ===
COPY . /app

# === Install Python dependencies ===
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# === Expose Streamlit default port ===
EXPOSE 8501

# === Run Streamlit app ===
CMD ["streamlit", "run", "app.py"]