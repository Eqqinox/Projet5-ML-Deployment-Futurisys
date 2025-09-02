"""
FastAPI Application principale - Projet Futurisys
API de prédiction d'attrition des employés avec modèle XGBoost
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn
import os
from contextlib import asynccontextmanager

from app.routers import health, predictions
from app.core.config import settings
from app.models.ml_model import MLModel

# Instance globale du modèle ML
ml_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application"""
    global ml_model
    
    # Startup: Chargement du modèle
    try:
        ml_model = MLModel()
        ml_model.load_model()
        app.state.ml_model = ml_model
        print("✅ Modèle XGBoost chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        raise e
    
    yield
    
    # Shutdown: Nettoyage si nécessaire
    print("🛑 Arrêt de l'application")

# Configuration FastAPI
app = FastAPI(
    title="Futurisys ML API - Prédiction d'Attrition",
    description="""
    API de Machine Learning pour prédire l'attrition des employés.
    
    ## Fonctionnalités
    
    * **Prédictions individuelles** : Prédire si un employé va quitter l'entreprise
    * **Prédictions batch** : Traiter plusieurs employés simultanément
    * **Explicabilité** : Comprendre les facteurs de décision
    * **Monitoring** : Suivi de la santé du modèle
    
    ## Modèle
    
    * **Algorithme** : XGBoost Classifier
    * **Variable cible** : a_quitte_l_entreprise (Oui/Non)
    * **Features** : 27 variables d'entrée (satisfaction, évaluation, démographie, etc.)
    """,
    version="1.0.0",
    contact={
        "name": "Futurisys ML Team",
        "email": "mounir.meknaci@gmail.com",
    },
    license_info={
        "name": "MIT License",
    },
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes principales
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])

@app.get("/", include_in_schema=False)
async def root():
    """Redirection vers la documentation"""
    return RedirectResponse(url="/docs")

@app.get("/info", tags=["Info"])
async def get_api_info():
    """Informations générales sur l'API"""
    return {
        "project": "Projet5 ML Deployment",
        "client": "Futurisys",
        "model_type": "XGBoost Classifier",
        "prediction_target": "Employee Attrition (a_quitte_l_entreprise)",
        "features_count": 27,
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "model_loaded": hasattr(app.state, 'ml_model') and app.state.ml_model is not None
    }

# Point d'entrée pour développement local
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )