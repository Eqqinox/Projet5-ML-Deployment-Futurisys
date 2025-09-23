"""
Gestionnaire de connexions PostgreSQL pour l'API Futurisys
Gestion des sessions de base de données et traçabilité des prédictions ML
"""

import logging    # Système de logs pour traçabilité
from typing import Optional, Dict, Any, Generator   # Type hints pour validation statique
from contextlib import contextmanager   # Décorateur pour context managers
from datetime import datetime   # Gestion des dates/heures
import uuid   # Génération d'identifiants uniques

from sqlalchemy import create_engine, text, func    # ORM SQLAlchemy pour PostgreSQL
from sqlalchemy.orm import sessionmaker, Session    # Gestion des sessions DB
from sqlalchemy.exc import SQLAlchemyError    # Exceptions spécifiques SQLAlchemy
from sqlalchemy.engine import Engine    # Type hint pour moteur DB

from app.core.config import settings    # Configuration centralisée (URL DB, etc.)
from app.database.models import (       # Modèles de données (tables)
    Employee, PredictionSession, PredictionRequest, 
    PredictionResult, ModelMetadata, APIAuditLog, Base
)

logger = logging.getLogger(__name__)    # Logger spécifique à ce module

class DatabaseManager:
    """
    Gestionnaire principal des connexions et opérations de base de données
    """
    
    def __init__(self):
        self.engine: Optional[Engine] = None    # Moteur SQLAlchemy (connexion principale)
        self.SessionLocal: Optional[sessionmaker] = None    # Factory pour créer des sessions
        self._is_initialized = False    # Flag d'état d'initialisation
    
    def initialize_database(self) -> bool:
        """
        Initialise la connexion à la base de données
        """
        try:
            # Création de l'engine SQLAlchemy avec pool de connexion
            self.engine = create_engine(
                settings.DATABASE_URL,    # URL de connexion depuis config
                pool_size=5,              # 5 connexions permanentes dans le pool
                max_overflow=10,          # 10 connexions supplémentaires si besoin
                pool_timeout=30,          # Timeout 30s pour obtenir une connexion
                pool_recycle=3600,        # Renouvelle les connexions chaque heure
                pool_pre_ping=True,       # Teste les connexions avant usage
                echo=settings.DEBUG       # Active les logs SQL en mode debug
            )
            
            # Test de connexion pour vérifier la disponibilité de PostgreSQL
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                logger.info(f"Connexion PostgreSQL établie: {version}")
            
            # Configuration des sessions
            self.SessionLocal = sessionmaker(
                bind=self.engine,     # Lie à notre engine PostgreSQL
                autocommit=False,     # Transactions manuelles (plus sûr)
                autoflush=False       # Flush manuel pour contrôler le timing
            )
            
            self._is_initialized = True
            logger.info("DatabaseManager initialisé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base: {e}")
            self._is_initialized = False
            raise e
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager pour obtenir une session de base de données
        """
        if not self._is_initialized:
            raise RuntimeError("DatabaseManager non initialisé")
        
        session = self.SessionLocal()   # Création d'une nouvelle session
        try:
            yield session               # Retourne la session au code appelant
            session.commit()            # Commit automatique si pas d'erreur
        except Exception as e:
            session.rollback()          # Annule toutes les modifications en cas d'erreur
            logger.error(f"Erreur de session de base de données: {e}")
            raise e
        finally:
            session.close()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Vérification de santé de la base de données
        """
        health_info = {
            "connection_ok": False,         # Connexion PostgreSQL fonctionnelle ?
            "tables_exist": False,          # Tables principales créées ?
            "employee_count": 0,            # Nombre d'employés en base
            "model_metadata_count": 0,      # Nombre de modèles enregistrés
            "last_check": datetime.now().isoformat()  # Timestamp du check
        }
        
        try:
            if not self._is_initialized:
                return health_info
                
            with self.get_session() as session:
                # Test de connexion basique
                session.execute(text("SELECT 1"))   # Requête minimale
                health_info["connection_ok"] = True
                
                # Vérification des tables principales
                tables_query = text("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('employees', 'prediction_sessions', 'prediction_requests', 'prediction_results')
                """)
                table_count = session.execute(tables_query).scalar()
                health_info["tables_exist"] = table_count >= 4    # Au moins 4 tables principales
                
                # Comptage des employés (Comptage des enregistrements pour vérifier l'intégrité)
                if health_info["tables_exist"]:
                    health_info["employee_count"] = session.query(Employee).count()
                    health_info["model_metadata_count"] = session.query(ModelMetadata).count()
                
        except Exception as e:
            logger.error(f"Erreur lors du health check: {e}")
            health_info["error"] = str(e)   # Inclut l'erreur dans le diagnostic
        
        return health_info
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Informations détaillées sur la base de données
        """
        if not self._is_initialized:
            return {"error": "Base de données non initialisée"}
        
        try:
            with self.get_session() as session:
                # Informations générales (version PostgreSQL)
                version_result = session.execute(text("SELECT version()"))
                db_version = version_result.scalar()
                
                # Comptages des tables (tous les types d'enregistrements)
                employee_count = session.query(Employee).count()
                session_count = session.query(PredictionSession).count()
                request_count = session.query(PredictionRequest).count()
                result_count = session.query(PredictionResult).count()
                
                # Informations sur le modèle ML actuel
                active_model = session.query(ModelMetadata).filter_by(is_active=True).first()
                
                return {
                    "database_version": db_version,
                    "tables": {     # Statistiques par table
                        "employees": employee_count,
                        "prediction_sessions": session_count,
                        "prediction_requests": request_count,
                        "prediction_results": result_count
                    },
                    "active_model": {
                        "name": active_model.model_name if active_model else None,
                        "version": active_model.version if active_model else None,
                        "algorithm": active_model.algorithm_type if active_model else None
                    } if active_model else None,
                    "traceability_enabled": True    # Confirmation que la traçabilité fonctionne
                }
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos DB: {e}")
            return {"error": str(e)}
    
    def verify_model_compatibility(self, ml_model) -> bool:
        """
        Vérifie la compatibilité entre le modèle ML et les métadonnées en base
        """
        try:
            with self.get_session() as session:
                # Récupération du modèle actif en base
                db_model = session.query(ModelMetadata).filter_by(is_active=True).first()
                
                if not db_model:
                    logger.warning("Aucun modèle actif trouvé en base de données")
                    return False
                
                # Récupération des métadonnées du modèle chargé en mémoire
                model_info = ml_model.get_model_info()
                
                # Vérifications de compatibilité critiques
                compatibility_checks = {
                    "version_match": db_model.version == model_info.get("version", "unknown"),
                    "threshold_match": abs(float(db_model.threshold_value) - model_info.get("threshold", 0.5)) < 0.001,
                    "algorithm_match": db_model.algorithm_type == model_info.get("model_type", "unknown")
                }
                
                all_compatible = all(compatibility_checks.values())
                
                if not all_compatible:
                    logger.warning(f"Incompatibilités détectées: {compatibility_checks}")
                else:
                    logger.info("Modèle ML compatible avec les métadonnées en base")
                
                return all_compatible
                
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de compatibilité: {e}")
            return False
    
    def create_prediction_session(self, session_type: str = "single", metadata: Dict = None) -> str:
        """
        Crée une nouvelle session de prédiction
        """
        try:
            with self.get_session() as db_session:
                session_obj = PredictionSession(
                    session_type=session_type,
                    status="pending",
                    session_metadata=metadata or {}
                )
                
                db_session.add(session_obj)   # Ajoute à la session SQLAlchemy
                db_session.flush()      # Pour obtenir l'ID (Force l'insertion pour récupérer l'ID)
                
                session_id = str(session_obj.session_id)    # Conversion UUID -> string
                logger.info(f"Session de prédiction créée: {session_id}")
                
                return session_id
                
        except Exception as e:
            logger.error(f"Erreur lors de la création de session: {e}")
            raise e
    
    def save_prediction_request(self, session_id: str, input_data: Dict, 
                               employee_id: Optional[int] = None) -> int:
        """
        Sauvegarde une requête de prédiction (input)
        """
        try:
            with self.get_session() as db_session:
                request_obj = PredictionRequest(
                    session_id=uuid.UUID(session_id),     # Conversion string -> UUID
                    employee_id=employee_id,              # Lien vers employé si existant
                    input_data=input_data,                # Données JSON complètes
                    request_source="api"                  # Source de la requête
                )
                
                db_session.add(request_obj)
                db_session.flush()                        # Récupère l'ID généré
                
                request_id = request_obj.request_id
                logger.debug(f"Requête de prédiction sauvegardée: {request_id}")
                
                return request_id
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de requête: {e}")
            raise e
    
    def save_prediction_result(self, request_id: int, prediction_result) -> int:
        """
        Sauvegarde un résultat de prédiction (output)
        """
        try:
            with self.get_session() as db_session:
                result_obj = PredictionResult(
                    request_id=request_id,
                    prediction=prediction_result.prediction,
                    probability_quit=float(prediction_result.probability_quit),
                    probability_stay=float(prediction_result.probability_stay),
                    confidence_level=prediction_result.confidence_level,
                    risk_factors=prediction_result.risk_factors,
                    model_version=prediction_result.model_version
                )
                
                db_session.add(result_obj)
                db_session.flush()
                
                result_id = result_obj.result_id
                logger.debug(f"Résultat de prédiction sauvegardé: {result_id}")
                
                return result_id
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de résultat: {e}")
            raise e
    
    def complete_prediction_session(self, session_id: str, total_predictions: int):
        """
        Marque une session de prédiction comme terminée
        """
        try:
            with self.get_session() as db_session:
                session_obj = db_session.query(PredictionSession).filter_by(
                    session_id=uuid.UUID(session_id)
                ).first()
                
                if session_obj:
                    session_obj.status = "completed"    # Statut final
                    session_obj.total_predictions = total_predictions
                    session_obj.completed_at = func.current_timestamp()     # Timestamp de fin
                    
                    logger.debug(f"Session {session_id} marquée comme terminée")
                else:
                    logger.warning(f"Session {session_id} non trouvée pour completion")
                    
        except Exception as e:
            logger.error(f"Erreur lors de la completion de session: {e}")
            raise e
    
    def get_employee_by_data(self, employee_data: Dict) -> Optional[Employee]:
        """
        Recherche un employé en base par ses caractéristiques
        (pour associer une prédiction à un employé existant)
        """
        try:
            with self.get_session() as session:
                # Recherche par caractéristiques uniques (âge + salaire + département)
                employee = session.query(Employee).filter(
                    Employee.age == employee_data.get('age'),
                    Employee.revenu_mensuel == employee_data.get('revenu_mensuel'),
                    Employee.departement == employee_data.get('departement'),
                    Employee.poste == employee_data.get('poste')
                ).first()
                
                return employee
                
        except Exception as e:
            logger.error(f"Erreur lors de la recherche d'employé: {e}")
            return None
    
    def get_prediction_history(self, limit: int = 100) -> list:
        """
        Récupère l'historique des prédictions
        """
        try:
            with self.get_session() as session:
                # Jointure pour récupérer toutes les informations
                query = session.query(
                    PredictionSession.session_id,
                    PredictionSession.session_type,
                    PredictionSession.started_at,
                    PredictionResult.prediction,
                    PredictionResult.probability_quit,
                    PredictionResult.confidence_level,
                    PredictionResult.model_version
                ).join(
                    PredictionRequest, PredictionSession.session_id == PredictionRequest.session_id
                ).join(
                    PredictionResult, PredictionRequest.request_id == PredictionResult.request_id
                ).order_by(
                    PredictionSession.started_at.desc()
                ).limit(limit)
                
                # Conversion en dictionnaires pour sérialisation JSON
                results = []
                for row in query.all():
                    results.append({
                        "session_id": str(row.session_id),
                        "session_type": row.session_type,
                        "started_at": row.started_at.isoformat(),
                        "prediction": row.prediction,
                        "probability_quit": float(row.probability_quit),
                        "confidence_level": row.confidence_level,
                        "model_version": row.model_version
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique: {e}")
            return []
    
    def close_connections(self):
        """
        Ferme les connexions à la base de données
        Appelé lors de l'arrêt de l'application
        """
        if self.engine:
            self.engine.dispose()     # Ferme toutes les connexions du pool
            logger.info("Connexions base de données fermées")
        
        self._is_initialized = False  # Remet l'état à non-initialisé