# Dockerfile pour Hugging Face Spaces - Projet Futurisys
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Variables d'environnement pour HF Spaces
ENV PYTHONPATH=/app
ENV PORT=7860

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app/ ./app/
COPY README.md .

# Créer le point d'entrée pour HF Spaces
COPY hf_app.py .

# Exposer le port requis par HF Spaces
EXPOSE 7860

# Commande pour démarrer l'application
CMD ["python", "hf_app.py"]