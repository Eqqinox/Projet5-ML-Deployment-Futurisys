"""
API FastAPI pour la prédiction d'attrition des employés - Futurisys
Version enrichie avec documentation OpenAPI complète
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn
import os
from contextlib import asynccontextmanager
import logging

from app.routers import health, predictions
from app.core.config import settings
from app.models.ml_model import MLModel
from app.database.connection import DatabaseManager
from app.middleware.prediction_logger import PredictionLoggerMiddleware

# Configuration du logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application FastAPI"""
    logger.info("🚀 Démarrage de l'API Futurisys ML...")
    
    # Initialisation de la base de données
    try:
        logger.info("📊 Initialisation PostgreSQL...")
        db_manager = DatabaseManager()
        db_manager.initialize_database()
        app.state.db_manager = db_manager
        logger.info("✅ PostgreSQL connecté")
    except Exception as e:
        logger.error(f"❌ Erreur PostgreSQL: {e}")
        app.state.db_manager = None
    
    # Chargement du modèle ML
    try:
        ml_model = MLModel()
        ml_model.load_model()
        app.state.ml_model = ml_model
        logger.info("✅ Modèle ML chargé")
    except Exception as e:
        logger.error(f"❌ Erreur modèle: {e}")
        app.state.ml_model = None
    
    yield
    
    # Nettoyage
    logger.info("🔄 Arrêt de l'API...")
    if hasattr(app.state, 'db_manager') and app.state.db_manager:
        app.state.db_manager.close_connections()
    app.state.ml_model = None
    logger.info("✅ Nettoyage terminé")

# Métadonnées pour organiser la documentation
tags_metadata = [
    {
        "name": "Health",
        "description": "Endpoints de monitoring et santé de l'API"
    },
    {
        "name": "Predictions", 
        "description": "Prédictions ML avec le modèle XGBoost"
    },
    {
        "name": "Data",
        "description": "Gestion des données PostgreSQL"
    },
    {
        "name": "Analytics",
        "description": "Statistiques et rapports"
    },
    {
        "name": "Info",
        "description": "Informations système"
    }
]


# Configuration FastAPI
app = FastAPI(
    title="Futurisys ML API - Prédiction d'Attrition",
    description="""
    API de Machine Learning pour prédire l'attrition des employés développée pour **Futurisys**.

    ## 🎯 Fonctionnalités

    * **Prédictions individuelles** : Analyser un employé spécifique
    * **Prédictions par lots** : Traiter jusqu'à 100 employés simultanément
    * **Validation des données** : Vérifier la conformité avant prédiction  
    * **Monitoring complet** : État de santé de l'API et du modèle
    * **Traçabilité** : Historique complet avec audit trail

    ## 🧠 Modèle XGBoost

    * **Version** : 1.0.0 (issu du Projet 4)
    * **Variables** : 27 features employé 
    * **Seuil optimal** : 0.514
    * **Performances** : Accuracy 85.88%, F1-Score 56.56%

    ## 📊 Architecture

    * **Backend** : FastAPI + PostgreSQL + Docker
    * **CI/CD** : GitHub Actions → Hugging Face Spaces
    * **Tests** : 85 tests automatisés (52% couverture)

    ---
    **Projet** : Formation Expert Data Science OpenClassrooms  
    **Développeur** : MMeknaci
    """,
    
    version="1.0.0",
    contact={
        "name": "Futurisys ML Team",
        "email": "mounir.meknaci@gmail.com",
    },
    license_info={
        "name": "MIT License",
    },
    openapi_tags=tags_metadata,
    openapi_url="/openapi.json", # Force la régénération
    docs_url="/docs",            # Force la régénération
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

# Middleware de traçabilité
app.add_middleware(PredictionLoggerMiddleware)

# Routes principales
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])

@app.get("/", include_in_schema=False)
async def root():
    """Redirection vers la documentation"""
    return RedirectResponse(url="/docs")

@app.get(
    "/info", 
    tags=["Info"],
    summary="Informations système et état de l'API",
    description="""
    Retourne les métadonnées complètes sur l'API Futurisys ML :
    
    - **État du modèle** : Vérification du chargement XGBoost
    - **Configuration** : Environnement et version déployée
    - **Endpoints** : Liste des fonctionnalités disponibles
    - **Monitoring** : Indicateurs de santé pour supervision
    
    Utilisé pour la validation de déploiement et le monitoring automatique.
    """,
    responses={
        200: {
            "description": "Informations récupérées avec succès",
            "content": {
                "application/json": {
                    "example": {
                        "project": "Projet5 ML Deployment",
                        "client": "Futurisys", 
                        "model_type": "XGBoost Classifier",
                        "version": "1.0.0",
                        "environment": "production",
                        "model_status": "✅ Modèle chargé et opérationnel",
                        "endpoints_available": [
                            "GET /health/ - Santé de l'API",
                            "POST /api/v1/predict/single - Prédiction individuelle"
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Erreur interne",
            "content": {
                "application/json": {
                    "example": {"detail": "Erreur système interne"}
                }
            }
        }
    }
)
async def get_api_info():
    """Informations générales sur l'API"""
    
    try:
        api_info = {
            "project": "Projet5 ML Deployment",
            "client": "Futurisys",
            "model_type": "XGBoost Classifier",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "endpoints_available": [
                "GET /health/ - Santé de l'API",
                "POST /api/v1/predict/single - Prédiction individuelle", 
                "POST /api/v1/predict/batch - Prédictions par lots",
                "POST /api/v1/predict/validate-input - Validation des données",
                "GET /api/v1/predict/supported-values - Valeurs acceptées",
                "GET /docs - Documentation Swagger"
            ]
        }
        
        # Vérification enrichie du modèle ML
        if hasattr(app.state, 'ml_model') and app.state.ml_model is not None:
            if app.state.ml_model.is_loaded:
                api_info["model_status"] = "✅ Modèle chargé et opérationnel"
                api_info["model_threshold"] = app.state.ml_model.threshold
                api_info["model_features"] = len(app.state.ml_model.final_column_names) if app.state.ml_model.final_column_names else "Unknown"
            else:
                api_info["model_status"] = "⚠️ Modèle présent mais non initialisé"
        else:
            api_info["model_status"] = "❌ Modèle non disponible"
        
        return api_info
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la récupération des informations système"
        )

# Routers pour PostgreSQL
try:
    from app.routers import data_management, analytics
    app.include_router(data_management.router, prefix="/api/v1/data", tags=["Data"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
except ImportError as e:
    logger.warning(f"⚠️ Routers optionnels non disponibles: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )