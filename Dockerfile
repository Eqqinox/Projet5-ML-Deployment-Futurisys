# Dockerfile pour Hugging Face Spaces - Version debug
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /code

# Variables d'environnement pour HF Spaces
ENV PYTHONPATH=/code
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

# Debug : Vérifier l'état avant copie
RUN echo "=== AVANT COPIE ==="
RUN ls -la /code/

# Copier TOUT le contenu (nouvelle approche)
COPY . /code/

# Debug : Vérifier l'état après copie
RUN echo "=== APRES COPIE ==="
RUN ls -la /code/
RUN echo "=== CONTENU APP ==="
RUN ls -la /code/app/ || echo "app/ directory not accessible"
RUN echo "=== FICHIERS PYTHON ==="
RUN find /code -name "*.py" | head -10

# Créer le dossier logs si nécessaire
RUN mkdir -p /code/logs

# Exposer le port requis par HF Spaces
EXPOSE 7860

# Commande pour démarrer l'application
CMD ["python", "hf_app.py"]