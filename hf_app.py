"""
Point d'entrée pour Hugging Face Spaces - Version production
"""
import uvicorn
import os
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )