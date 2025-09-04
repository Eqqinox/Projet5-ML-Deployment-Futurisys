"""
Gestionnaire de connexions PostgreSQL pour l'API Futurisys
Gestion des sessions de base de données et traçabilité des prédictions ML
"""

import logging
from typing import Optional, Dict, Any, Generator
from contextlib import contextmanager
from datetime import datetime
import uuid

from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.database.models import (
    Employee, PredictionSession, PredictionRequest, 
    PredictionResult, ModelMetadata, APIAuditLog, Base
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Gestionnaire principal des connexions et opérations de base de données
    """
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._is_initialized = False
    
    def initialize_database(self) -> bool:
        """
        Initialise la connexion à la base de données
        """
        try:
            # Création de l'engine SQLAlchemy
            self.engine = create_engine(
                settings.DATABASE_URL,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=settings.DEBUG  # Logs SQL en mode debug
            )
            
            # Test de connexion
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                logger.info(f"Connexion PostgreSQL établie: {version}")
            
            # Configuration des sessions
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
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
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur de session de base de données: {e}")
            raise e
        finally:
            session.close()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Vérification de santé de la base de données
        """
        health_info = {
            "connection_ok": False,
            "tables_exist": False,
            "employee_count": 0,
            "model_metadata_count": 0,
            "last_check": datetime.now().isoformat()
        }
        
        try:
            if not self._is_initialized:
                return health_info
                
            with self.get_session() as session:
                # Test de connexion basique
                session.execute(text("SELECT 1"))
                health_info["connection_ok"] = True
                
                # Vérification des tables principales
                tables_query = text("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('employees', 'prediction_sessions', 'prediction_requests', 'prediction_results')
                """)
                table_count = session.execute(tables_query).scalar()
                health_info["tables_exist"] = table_count >= 4
                
                # Comptage des employés
                if health_info["tables_exist"]:
                    health_info["employee_count"] = session.query(Employee).count()
                    health_info["model_metadata_count"] = session.query(ModelMetadata).count()
                
        except Exception as e:
            logger.error(f"Erreur lors du health check: {e}")
            health_info["error"] = str(e)
        
        return health_info
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Informations détaillées sur la base de données
        """
        if not self._is_initialized:
            return {"error": "Base de données non initialisée"}
        
        try:
            with self.get_session() as session:
                # Informations générales
                version_result = session.execute(text("SELECT version()"))
                db_version = version_result.scalar()
                
                # Comptages des tables
                employee_count = session.query(Employee).count()
                session_count = session.query(PredictionSession).count()
                request_count = session.query(PredictionRequest).count()
                result_count = session.query(PredictionResult).count()
                
                # Informations sur le modèle actuel
                active_model = session.query(ModelMetadata).filter_by(is_active=True).first()
                
                return {
                    "database_version": db_version,
                    "tables": {
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
                    "traceability_enabled": True
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
                
                # Vérifications de compatibilité
                model_info = ml_model.get_model_info()
                
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
                
                db_session.add(session_obj)
                db_session.flush()  # Pour obtenir l'ID
                
                session_id = str(session_obj.session_id)
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
                    session_id=uuid.UUID(session_id),
                    employee_id=employee_id,
                    input_data=input_data,
                    request_source="api"
                )
                
                db_session.add(request_obj)
                db_session.flush()
                
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
                    session_obj.status = "completed"
                    session_obj.total_predictions = total_predictions
                    session_obj.completed_at = func.current_timestamp()
                    
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
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Connexions base de données fermées")
        
        self._is_initialized = False