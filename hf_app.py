"""
Point d'entrée pour Hugging Face Spaces - Version production
"""
import sys
import os
import uvicorn

# Ajouter le répertoire courant au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Maintenant importer l'app
try:
    from app.main import app
    print("✅ Module app.main importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print(f"📁 Répertoire courant: {os.getcwd()}")
    print(f"🐍 PYTHONPATH: {sys.path}")
    print(f"📋 Contenu du répertoire:")
    for item in os.listdir('.'):
        print(f"  - {item}")
    if os.path.exists('app'):
        print(f"📋 Contenu de app/:")
        for item in os.listdir('app'):
            print(f"  - app/{item}")
    raise

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )