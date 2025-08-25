"""
Point d'entrée pour Hugging Face Spaces - Projet Futurisys
API FastAPI minimale pour tester le déploiement HF Spaces
"""

import uvicorn
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# API minimale pour tester le déploiement
app = FastAPI(
    title="Futurisys ML API",
    description="API de Machine Learning pour le client Futurisys",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "Futurisys ML API - Déploiement HF Spaces OK"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "futurisys-ml-api"}

@app.get("/info")
async def info():
    return {
        "project": "Projet5 ML Deployment",
        "client": "Futurisys",
        "stack": ["FastAPI", "XGBoost", "PostgreSQL"],
        "environment": "Hugging Face Spaces"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Port requis par HF Spaces
    
    # Configuration pour HF Spaces
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )