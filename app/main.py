"""
FastAPI Application principale - Projet Futurisys
API de prédiction d'attrition des employés avec modèle XGBoost et traçabilité PostgreSQL
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

# Instances globales
ml_model = None
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application avec BDD"""
    global ml_model, db_manager
    
    # Startup: Initialisation des composants
    try:
        logger.info("🚀 Initialisation de l'application Futurisys ML")
        
        # 1. Initialisation de la base de données
        logger.info("📊 Initialisation de la connexion PostgreSQL...")
        db_manager = DatabaseManager()
        db_manager.initialize_database()
        app.state.db_manager = db_manager
        logger.info("✅ Connexion PostgreSQL établie")
        
        # 2. Chargement du modèle ML
        logger.info("🤖 Chargement du modèle XGBoost...")
        ml_model = MLModel()
        ml_model.load_model()
        app.state.ml_model = ml_model
        logger.info("✅ Modèle XGBoost chargé avec succès")
        
        # 3. Vérification de la cohérence BDD ↔ Modèle
        logger.info("🔍 Vérification de la cohérence BDD ↔ Modèle...")
        db_manager.verify_model_compatibility(ml_model)
        logger.info("✅ Cohérence BDD ↔ Modèle vérifiée")
        
        logger.info("🎯 Application Futurisys ML prête !")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        # Ne pas faire crash l'app, mais logger l'erreur
        # L'application pourra fonctionner en mode dégradé
        if 'ml_model' in locals():
            app.state.ml_model = ml_model
        if 'db_manager' in locals():
            app.state.db_manager = db_manager
    
    yield
    
    # Shutdown: Nettoyage des ressources
    logger.info("🛑 Arrêt de l'application")
    if db_manager:
        db_manager.close_connections()
        logger.info("🔌 Connexions base de données fermées")

# Configuration FastAPI
app = FastAPI(
    title="Futurisys ML API - Prédiction d'Attrition",
    description="""
    API de Machine Learning pour prédire l'attrition des employés avec traçabilité complète.
    
    ## Fonctionnalités
    
    * **Prédictions individuelles** : Prédire si un employé va quitter l'entreprise
    * **Prédictions batch** : Traiter plusieurs employés simultanément
    * **Traçabilité complète** : Tous les inputs/outputs sauvegardés en base PostgreSQL
    * **Explicabilité** : Comprendre les facteurs de décision
    * **Analytics** : Consultation de l'historique et des statistiques
    * **Monitoring** : Suivi de la santé du modèle et de la base
    
    ## Modèle
    
    * **Algorithme** : XGBoost Classifier (v1.0.0)
    * **Variable cible** : a_quitte_l_entreprise (Oui/Non)
    * **Features** : 27 variables d'entrée (satisfaction, évaluation, démographie, etc.)
    * **Dataset** : 1470 employés du Projet 4
    * **Seuil optimal** : 0.514
    
    ## Base de données
    
    * **SGBD** : PostgreSQL 16+
    * **Tables** : 6 tables avec traçabilité complète
    * **Workflow** : API → BDD → Modèle → BDD → Réponse
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

# Middleware de traçabilité des prédictions
app.add_middleware(PredictionLoggerMiddleware)

# Routes principales
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])

# Nouveaux routers pour la gestion des données
try:
    from app.routers import data_management, analytics
    app.include_router(data_management.router, prefix="/api/v1/data", tags=["Data Management"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
except ImportError:
    logger.warning("⚠️ Routers data_management ou analytics non disponibles")

@app.get("/", include_in_schema=False)
async def root():
    """Redirection vers la documentation"""
    return RedirectResponse(url="/docs")

@app.get("/info", tags=["Info"])
async def get_api_info():
    """Informations générales sur l'API et la base de données"""
    
    # Informations de base
    api_info = {
        "project": "Projet5 ML Deployment",
        "client": "Futurisys",
        "model_type": "XGBoost Classifier",
        "prediction_target": "Employee Attrition (a_quitte_l_entreprise)",
        "features_count": 27,
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "model_loaded": hasattr(app.state, 'ml_model') and app.state.ml_model is not None,
        "database_connected": hasattr(app.state, 'db_manager') and app.state.db_manager is not None
    }
    
    # Informations sur la base de données (si disponible)
    if hasattr(app.state, 'db_manager') and app.state.db_manager:
        try:
            db_info = app.state.db_manager.get_database_info()
            api_info.update({
                "database_info": db_info,
                "traceability_enabled": True
            })
        except Exception as e:
            api_info["database_error"] = str(e)
    else:
        api_info["traceability_enabled"] = False
    
    return api_info

@app.get("/status", tags=["Info"])
async def get_system_status():
    """Status détaillé du système (modèle + base de données)"""
    
    status = {
        "api_status": "healthy",
        "timestamp": "current_timestamp_will_be_added",
        "components": {}
    }
    
    # Status du modèle ML
    if hasattr(app.state, 'ml_model') and app.state.ml_model:
        try:
            model_health = app.state.ml_model.health_check()
            status["components"]["ml_model"] = {
                "status": "healthy" if all(model_health.values()) else "degraded",
                "details": model_health
            }
        except Exception as e:
            status["components"]["ml_model"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        status["components"]["ml_model"] = {"status": "not_loaded"}
    
    # Status de la base de données
    if hasattr(app.state, 'db_manager') and app.state.db_manager:
        try:
            db_health = app.state.db_manager.health_check()
            status["components"]["database"] = {
                "status": "healthy" if db_health["connection_ok"] else "error",
                "details": db_health
            }
        except Exception as e:
            status["components"]["database"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        status["components"]["database"] = {"status": "not_connected"}
    
    # Status global
    component_statuses = [comp["status"] for comp in status["components"].values()]
    if "error" in component_statuses:
        status["api_status"] = "error"
    elif "degraded" in component_statuses or "not_loaded" in component_statuses:
        status["api_status"] = "degraded"
    
    return status

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire global d'exceptions avec logging"""
    logger.error(f"Erreur non gérée: {exc}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail={
            "error": "Internal Server Error",
            "message": "Une erreur interne s'est produite",
            "request_path": str(request.url),
            "type": type(exc).__name__
        }
    )

# Point d'entrée pour développement local
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )