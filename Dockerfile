# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . /app

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose ports for Streamlit and MLflow
EXPOSE 8501
EXPOSE 5001

# Launch Streamlit app
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]