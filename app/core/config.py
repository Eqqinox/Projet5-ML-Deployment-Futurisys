"""
Configuration de l'application FastAPI
Gestion des variables d'environnement et paramètres
"""

#### Pydantic v2 pour charger automatiquement les variables d’environnement avec validation de type. ####
from pydantic_settings import BaseSettings
from typing import Optional     #certaines variables peuvent être None
import os

class Settings(BaseSettings):       #centralise toute la configuration de l'app
    """
    Configuration de l'application basée sur les variables d'environnement
    """
    
    # Configuration API
    API_HOST: str = "0.0.0.0"               #adresse d’écoute du serveur
    API_PORT: int = 8000                    #port d’écoute
    DEBUG: bool = True                      #active le mode debug de FastAPI
    SECRET_KEY: str = "votre-clé-secrète-très-longue-et-complexe-12345"
    
    # Configuration Base de données
    DATABASE_URL: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    # Configuration ML Model
    MODEL_PATH: str = "app/models/trained_model.pkl"
    MODEL_VERSION: str = "1.0.0"
    
    # Configuration Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # Environnement
    ENVIRONMENT: str = "development"
    
    # Configuration de performance
    MAX_BATCH_SIZE: int = 100 #nombre max d’éléments à traiter en une requête
    PREDICTION_TIMEOUT: int = 30  #secondes
    
    # Configuration sécurité (pour plus tard)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256" #algo de signature
    
    class Config:
        env_file = ".env" #charge les variables depuis un fichier .env à la racine
        env_file_encoding = "utf-8"
        case_sensitive = True

# Instance globale des paramètres
settings = Settings()

# Configuration de logging
LOGGING_CONFIG = {     #Définit les formats de log (default, detailed)
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s",
        },
    },
    "handlers": {       #Où envoyer les logs (console, file)
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "detailed",
            "filename": settings.LOG_FILE,
            "mode": "a",
        },
    },
    "loggers": {        #Associe des handlers à des loggers spécifiques
        "": {  # root logger
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
        "app": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}