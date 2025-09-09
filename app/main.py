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
    
    # 1. Initialisation de la base de données
    try:
        logger.info("📊 Initialisation PostgreSQL...")
        db_manager = DatabaseManager()
        db_manager.initialize_database()
        app.state.db_manager = db_manager
        logger.info("✅ PostgreSQL connecté")
    except Exception as e:
        logger.error(f"❌ Erreur PostgreSQL: {e}")
        app.state.db_manager = None
    
    # 2. Chargement du modèle ML
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

# Configuration FastAPI
app = FastAPI(
    title="Futurisys ML API - Prédiction d'Attrition",
    description="""
    API de Machine Learning pour prédire l'attrition des employés.

    ## Fonctionnalités

    * **Prédictions individuelles** : Prédire si un employé va quitter l'entreprise
    * **Validation des données** : Vérifier la conformité des données d'entrée  
    * **Documentation des valeurs** : Consulter les valeurs acceptées
    * **Monitoring** : Vérifier l'état de santé de l'API et du modèle

    ## Modèle

    * **Algorithme** : XGBoost Classifier (v1.0.0)
    * **Variable cible** : a_quitte_l_entreprise (Oui/Non)  
    * **Features** : 27 variables d'entrée
    * **Seuil optimal** : 0.514
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

# Middleware de traçabilité
app.add_middleware(PredictionLoggerMiddleware)

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
    
    api_info = {
        "project": "Projet5 ML Deployment",
        "client": "Futurisys",
        "model_type": "XGBoost Classifier",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "endpoints_available": [
            "GET /health/ - Santé de l'API",
            "POST /api/v1/predict/single - Prédiction individuelle", 
            "POST /api/v1/predict/validate-input - Validation des données",
            "GET /api/v1/predict/supported-values - Valeurs acceptées",
            "GET /docs - Documentation Swagger"
        ]
    }
    
    # Vérifier si le modèle est chargé
    if hasattr(app.state, 'ml_model') and app.state.ml_model is not None:
        api_info["model_status"] = "✅ Modèle chargé et opérationnel"
    else:
        api_info["model_status"] = "❌ Modèle non disponible"
    
    return api_info

# Routers pour PostgreSQL
try:
    from app.routers import data_management, analytics
    app.include_router(data_management.router, prefix="/api/v1/data", tags=["Data"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
except ImportError as e:
    logger.warning(f"⚠️ Routers optionnels non disponibles: {e}")