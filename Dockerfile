# Dockerfile pour Hugging Face Spaces - Projet Futurisys
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

# Copier TOUT le contenu du projet (plus robuste)
COPY . .

# Créer le dossier logs si nécessaire
RUN mkdir -p logs

# Vérifier que la structure est bien copiée (debug)
RUN ls -la app/ || echo "app directory not found"

# Exposer le port requis par HF Spaces
EXPOSE 7860

# Commande pour démarrer l'application
CMD ["python", "hf_app.py"]