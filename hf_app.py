"""
Point d'entrée pour Hugging Face Spaces - Version production
"""
import uvicorn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )