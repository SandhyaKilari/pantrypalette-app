## 🐳 Docker Build & Deployment Guide

This guide outlines how the Docker images for the PantryPalette project were built and pushed to Docker Hub using `docker buildx` with `linux/amd64` compatibility.

### 📦 Docker Images

Two Docker images are used in the project:

1. **MLflow Tracking Server**  
   Hosts the experiment tracking and model registry backend.

2. **Streamlit Frontend App**  
   Provides the user interface for ingredient-based recipe recommendations.

### 🔧 Build & Push Commands

#### 1. MLflow Image

```bash
docker buildx build --platform linux/amd64   -f Dockerfile.mlflow   -t sandhyakilari/pantrypalette-mlflow:latest   --push .
```

#### 2. Streamlit App Image

```bash
docker buildx build --platform linux/amd64   -f Dockerfile.streamlit   -t sandhyakilari/pantrypalette:latest   --push .
```

### 📝 Notes

- `--platform linux/amd64`: Ensures cross-platform compatibility (e.g., AWS EC2, GitHub Actions runners)
- `--push`: Immediately pushes the image to Docker Hub after building
- Make sure `Dockerfile.mlflow` and `Dockerfile.streamlit` are in your build context

### 🔗 Docker Hub Links

- **Streamlit App**: [sandhyakilari/pantrypalette](https://hub.docker.com/r/sandhyakilari/pantrypalette)  
- **MLflow Server**: [sandhyakilari/pantrypalette-mlflow](https://hub.docker.com/r/sandhyakilari/pantrypalette-mlflow)

### ✅ Example Deployment (Docker Compose)

Both containers can be run locally or on a cloud VM using:

```bash
docker-compose up -d
```

🔗 **For detailed deployment steps and system architecture, refer to the** [PantryPalette_EC2_Docker_Deploy.md](https://github.com/sandhyakilari/pantrypalette-app/blob/main/PantryPalette_Documentation/PantryPalette_EC2_Docker_Deploy.md) **guide.**
