"""
Modèles SQLAlchemy pour l'API Futurisys
Réutilisation des modèles de database/python/create_db.py avec adaptations pour l'API
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Numeric, Text, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET, ARRAY
import uuid
from datetime import datetime

Base = declarative_base()

class Employee(Base):
    """
    Modèle Employee - Dataset Projet 4 avec 1470 employés
    """
    __tablename__ = 'employees'
    
    # Clé primaire
    employee_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Variables de satisfaction (1-4)
    satisfaction_employee_environnement = Column(Integer, nullable=False)
    satisfaction_employee_nature_travail = Column(Integer, nullable=False)
    satisfaction_employee_equipe = Column(Integer, nullable=False)
    satisfaction_employee_equilibre_pro_perso = Column(Integer, nullable=False)
    
    # Variables d'évaluation (1-4)
    note_evaluation_precedente = Column(Integer, nullable=False)
    note_evaluation_actuelle = Column(Integer, nullable=False)
    
    # Variables hiérarchiques
    niveau_hierarchique_poste = Column(Integer, nullable=False)
    
    # Variables binaires
    heure_supplementaires = Column(String(5), nullable=False)
    
    # Variable d'augmentation
    augementation_salaire_precedente = Column(Numeric(6,4), nullable=False)
    
    # Variables démographiques
    age = Column(Integer, nullable=False)
    genre = Column(String(5), nullable=False)
    revenu_mensuel = Column(Integer, nullable=False)
    statut_marital = Column(String(20), nullable=False)
    
    # Variables organisationnelles
    departement = Column(String(30), nullable=False)
    poste = Column(String(50), nullable=False)
    
    # Variables d'expérience
    nombre_experiences_precedentes = Column(Integer, nullable=False)
    annee_experience_totale = Column(Integer, nullable=False)
    annees_dans_l_entreprise = Column(Integer, nullable=False)
    annees_dans_le_poste_actuel = Column(Integer, nullable=False)
    annees_depuis_la_derniere_promotion = Column(Integer, nullable=False)
    annes_sous_responsable_actuel = Column(Integer, nullable=False)
    
    # Variables de formation
    nombre_participation_pee = Column(Integer, nullable=False)
    nb_formations_suivies = Column(Integer, nullable=False)
    
    # Variables géographiques
    distance_domicile_travail = Column(Integer, nullable=False)
    
    # Variables d'éducation
    niveau_education = Column(Integer, nullable=False)
    domaine_etude = Column(String(50), nullable=False)
    
    # Fréquence de déplacement
    frequence_deplacement = Column(String(20), nullable=False)
    
    # Variable cible
    a_quitte_l_entreprise = Column(String(5), nullable=False)
    
    # Métadonnées
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relations
    prediction_requests = relationship("PredictionRequest", back_populates="employee")
    
    def to_dict(self):
        """Convertit l'employé en dictionnaire pour l'API"""
        return {
            "employee_id": self.employee_id,
            "satisfaction_employee_environnement": self.satisfaction_employee_environnement,
            "satisfaction_employee_nature_travail": self.satisfaction_employee_nature_travail,
            "satisfaction_employee_equipe": self.satisfaction_employee_equipe,
            "satisfaction_employee_equilibre_pro_perso": self.satisfaction_employee_equilibre_pro_perso,
            "note_evaluation_precedente": self.note_evaluation_precedente,
            "note_evaluation_actuelle": self.note_evaluation_actuelle,
            "niveau_hierarchique_poste": self.niveau_hierarchique_poste,
            "heure_supplementaires": self.heure_supplementaires,
            "augementation_salaire_precedente": float(self.augementation_salaire_precedente),
            "age": self.age,
            "genre": self.genre,
            "revenu_mensuel": self.revenu_mensuel,
            "statut_marital": self.statut_marital,
            "departement": self.departement,
            "poste": self.poste,
            "nombre_experiences_precedentes": self.nombre_experiences_precedentes,
            "annee_experience_totale": self.annee_experience_totale,
            "annees_dans_l_entreprise": self.annees_dans_l_entreprise,
            "annees_dans_le_poste_actuel": self.annees_dans_le_poste_actuel,
            "annees_depuis_la_derniere_promotion": self.annees_depuis_la_derniere_promotion,
            "annes_sous_responsable_actuel": self.annes_sous_responsable_actuel,
            "nombre_participation_pee": self.nombre_participation_pee,
            "nb_formations_suivies": self.nb_formations_suivies,
            "distance_domicile_travail": self.distance_domicile_travail,
            "niveau_education": self.niveau_education,
            "domaine_etude": self.domaine_etude,
            "frequence_deplacement": self.frequence_deplacement,
            "a_quitte_l_entreprise": self.a_quitte_l_entreprise,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class PredictionSession(Base):
    """Sessions de prédiction (single/batch)"""
    __tablename__ = 'prediction_sessions'
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_type = Column(String(10), nullable=False)
    total_predictions = Column(Integer, default=0)
    status = Column(String(20), nullable=False, default='pending')
    started_at = Column(DateTime, default=func.current_timestamp())
    completed_at = Column(DateTime)
    error_message = Column(Text)
    session_metadata = Column(JSONB)
    
    # Relations
    prediction_requests = relationship("PredictionRequest", back_populates="session", cascade="all, delete-orphan")
    audit_logs = relationship("APIAuditLog", back_populates="session")
    
    def to_dict(self):
        """Convertit la session en dictionnaire pour l'API"""
        return {
            "session_id": str(self.session_id),
            "session_type": self.session_type,
            "total_predictions": self.total_predictions,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "session_metadata": self.session_metadata
        }

class PredictionRequest(Base):
    """Inputs du modèle ML"""
    __tablename__ = 'prediction_requests'
    
    request_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('prediction_sessions.session_id', ondelete='CASCADE'), nullable=False)
    employee_id = Column(Integer, ForeignKey('employees.employee_id', ondelete='SET NULL'))
    input_data = Column(JSONB, nullable=False)
    request_source = Column(String(20), nullable=False, default='api')
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    session = relationship("PredictionSession", back_populates="prediction_requests")
    employee = relationship("Employee", back_populates="prediction_requests")
    result = relationship("PredictionResult", back_populates="request", uselist=False, cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convertit la requête en dictionnaire pour l'API"""
        return {
            "request_id": self.request_id,
            "session_id": str(self.session_id),
            "employee_id": self.employee_id,
            "input_data": self.input_data,
            "request_source": self.request_source,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class PredictionResult(Base):
    """Outputs du modèle ML"""
    __tablename__ = 'prediction_results'
    
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(Integer, ForeignKey('prediction_requests.request_id', ondelete='CASCADE'), nullable=False)
    prediction = Column(String(5), nullable=False)
    probability_quit = Column(Numeric(6,4), nullable=False)
    probability_stay = Column(Numeric(6,4), nullable=False)
    confidence_level = Column(String(10), nullable=False)
    risk_factors = Column(ARRAY(Text))
    model_version = Column(String(20), nullable=False)
    processing_time_ms = Column(Numeric(10,2))
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    request = relationship("PredictionRequest", back_populates="result")
    
    def to_dict(self):
        """Convertit le résultat en dictionnaire pour l'API"""
        return {
            "result_id": self.result_id,
            "request_id": self.request_id,
            "prediction": self.prediction,
            "probability_quit": float(self.probability_quit),
            "probability_stay": float(self.probability_stay),
            "confidence_level": self.confidence_level,
            "risk_factors": self.risk_factors,
            "model_version": self.model_version,
            "processing_time_ms": float(self.processing_time_ms) if self.processing_time_ms else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ModelMetadata(Base):
    """Métadonnées et versioning des modèles ML"""
    __tablename__ = 'model_metadata'
    
    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False, unique=True)
    algorithm_type = Column(String(50), default='XGBoost')
    threshold_value = Column(Numeric(6,4), default=0.5)
    performance_metrics = Column(JSONB)
    feature_importance = Column(JSONB)
    model_file_path = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    deprecated_at = Column(DateTime)
    
    def to_dict(self):
        """Convertit les métadonnées en dictionnaire pour l'API"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "algorithm_type": self.algorithm_type,
            "threshold_value": float(self.threshold_value),
            "performance_metrics": self.performance_metrics,
            "feature_importance": self.feature_importance,
            "model_file_path": self.model_file_path,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None
        }

class APIAuditLog(Base):
    """Audit complet des appels API"""
    __tablename__ = 'api_audit_logs'
    
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('prediction_sessions.session_id', ondelete='SET NULL'))
    endpoint_called = Column(String(100), nullable=False)
    http_method = Column(String(10), nullable=False)
    client_ip = Column(INET)
    user_agent = Column(Text)
    request_headers = Column(JSONB)
    request_payload = Column(JSONB)
    response_status_code = Column(Integer, nullable=False)
    response_payload = Column(JSONB)
    response_time_ms = Column(Numeric(10,2))
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    session = relationship("PredictionSession", back_populates="audit_logs")
    
    def to_dict(self):
        """Convertit le log d'audit en dictionnaire pour l'API"""
        return {
            "log_id": self.log_id,
            "session_id": str(self.session_id) if self.session_id else None,
            "endpoint_called": self.endpoint_called,
            "http_method": self.http_method,
            "client_ip": str(self.client_ip) if self.client_ip else None,
            "user_agent": self.user_agent,
            "request_headers": self.request_headers,
            "request_payload": self.request_payload,
            "response_status_code": self.response_status_code,
            "response_payload": self.response_payload,
            "response_time_ms": float(self.response_time_ms) if self.response_time_ms else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }