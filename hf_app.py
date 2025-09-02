"""
Point d'entrée pour Hugging Face Spaces - Version production
"""
import uvicorn
import os
import sys

# Ajouter le sous-dossier app au chemin de recherche
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

print("Chemin sys.path:", sys.path)  # Pour déboguer
print("Contenu de /app:", os.listdir(os.path.dirname(__file__)))  # Pour lister les fichiers

from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )