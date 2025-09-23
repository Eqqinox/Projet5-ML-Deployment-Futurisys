"""
Configuration de l'application FastAPI
Gestion des variables d'environnement et paramètres
"""

#### Pydantic v2 pour charger automatiquement les variables d’environnement avec validation de type. ####
from pydantic_settings import BaseSettings  # Classe de base pour configuration avec validation automatique
from typing import Optional     # Type hint pour variables optionnelles (peuvent être None)
import os                       # Module système pour accès aux variables d'environnement

class Settings(BaseSettings):       #centralise toute la configuration de l'app
    """
    Configuration de l'application basée sur les variables d'environnement
    """
    
    # Configuration API
    API_HOST: str = "0.0.0.0"               #adresse d’écoute du serveur (0.0.0.0 = toutes les interfaces)
    API_PORT: int = 8000                    #port TCP d’écoute
    DEBUG: bool = True                      #active le mode debug de FastAPI
    SECRET_KEY: str = "clé-secrète" # A CHANGER
    
    # Configuration Base de données - Variables d'environnement (.env) - PRIORITÉ HAUTE
    # Les variables documentent quelles configurations sont attendues, même si les valeurs viennent du .env.
    # Si .env est corrompu ou manquant, les valeurs par défaut évitent un crash total
    DATABASE_URL: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432           # Valide que .env contient un nombre
    
    # Configuration ML Model
    MODEL_PATH: str = "app/models/trained_model.pkl"
    MODEL_VERSION: str = "1.0.0"
    
    # Configuration Logging
    LOG_LEVEL: str = "INFO"     # Niveau de log (DEBUG, INFO, WARNING, ERROR)
    LOG_FILE: str = "logs/app.log"
    
    # Environnement
    ENVIRONMENT: str = "development" #Valeur par défaut
    
    # Configuration de performance
    MAX_BATCH_SIZE: int = 100       #nombre max d’éléments à traiter en une requête
    PREDICTION_TIMEOUT: int = 30    #secondes
    
    # Configuration sécurité (pour plus tard)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"    #algo de signature des tokens
    
    class Config:
        env_file = ".env"               #charge les variables depuis le fichier .env à la racine
        env_file_encoding = "utf-8"     # Encodage du fichier .env
        case_sensitive = True           

# Instance globale des paramètres
settings = Settings()               # Charge automatiquement depuis .env + variables système

# Configuration de logging (dictionnaire Python standard)
LOGGING_CONFIG = {     # Configuration complète du système de logging
    "version": 1,
    "disable_existing_loggers": False,      # Conserve les loggers existants
    "formatters": {             # Définit les formats de messages de log
        "default": {            # Format simple pour la console            
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {           # Format détaillé pour les fichiers
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
        "file": {       # Handler pour sauvegarde fichier
            "class": "logging.FileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "detailed",
            "filename": settings.LOG_FILE,
            "mode": "a",
        },
    },
    "loggers": {        #Associe des handlers à des loggers spécifiques
        "": {           # root logger (logger par défaut)
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
        "app": {        # Logger pour l'app
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn": {    # Logger pour le serveur web
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}