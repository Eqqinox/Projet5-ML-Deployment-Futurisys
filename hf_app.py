"""
Point d'entrée pour Hugging Face Spaces - Version production
"""
import uvicorn
import os
import sys

# On ajoute le dossier racine du projet au PYTHONPATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("Chemin sys.path:", sys.path)

from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )