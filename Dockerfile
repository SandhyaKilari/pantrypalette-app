FROM python:3.10-slim

WORKDIR /app

# Copy everything from project root into container
COPY . /app

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port used by Streamlit
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "UI/app.py", "--server.port=8501", "--server.enableCORS=false"]