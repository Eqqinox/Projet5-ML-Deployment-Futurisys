"""
Modèles SQLAlchemy pour l'API Futurisys
Définition des tables PostgreSQL basée sur le dataset du projet 4 (1470 employés)
Architecture unifié pour l'import de données et l'API de prédiction ML
"""

# Imports SQLAlchemy pour définition des tables et relations
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Numeric, Text, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base   # Classe de base pour tous les modèles
from sqlalchemy.orm import relationship                   # Définition des relations entre tables
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET, ARRAY # Types spécifiques PostgreSQL
import uuid   # Génération d'identifiants uniques
from datetime import datetime   # Gestion des dates/heures

Base = declarative_base()     # Classe de base héritée par tous les modèles

class Employee(Base):
    """
    Modèle Employee - Dataset Projet 4 avec 1470 employés
    """
    __tablename__ = 'employees'
    
    # Clé primaire auto-incrémentée
    employee_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Variables de satisfaction (échelle 1-4 du Projet 4)
    satisfaction_employee_environnement = Column(Integer, nullable=False)
    satisfaction_employee_nature_travail = Column(Integer, nullable=False)
    satisfaction_employee_equipe = Column(Integer, nullable=False)
    satisfaction_employee_equilibre_pro_perso = Column(Integer, nullable=False)
    
    # Variables d'évaluation (échelle 1-4 du Projet 4)
    note_evaluation_precedente = Column(Integer, nullable=False)
    note_evaluation_actuelle = Column(Integer, nullable=False)
    
    # Variables hiérarchiques
    niveau_hierarchique_poste = Column(Integer, nullable=False)
    
    # Variables binaires (encodées en string pour cohérence avec dataset)
    heure_supplementaires = Column(String(5), nullable=False)   # "Oui" ou "Non"
    
    # Variable d'augmentation (pourcentage décimal)
    augementation_salaire_precedente = Column(Numeric(6,4), nullable=False) # Ex: 0.15 pour 15%
    
    # Variables démographiques
    age = Column(Integer, nullable=False)               # Âge en années
    genre = Column(String(5), nullable=False)           # "Homme" ou "Femme"
    revenu_mensuel = Column(Integer, nullable=False)    # Salaire mensuel en euros
    statut_marital = Column(String(20), nullable=False) # "Célibataire", "Marié(e)", etc.
    
    # Variables organisationnelles
    departement = Column(String(30), nullable=False)    # "Commercial", "Consulting", etc.
    poste = Column(String(50), nullable=False)          # "Manager", "Consultant", etc.
    
    # Variables d'expérience professionnelle
    nombre_experiences_precedentes = Column(Integer, nullable=False)    # Nb d'emplois précédents
    annee_experience_totale = Column(Integer, nullable=False)           # Total années d'expérience
    annees_dans_l_entreprise = Column(Integer, nullable=False)          # Ancienneté entreprise actuelle
    annees_dans_le_poste_actuel = Column(Integer, nullable=False)       # Ancienneté poste actuel
    annees_depuis_la_derniere_promotion = Column(Integer, nullable=False) # Dernière promotion
    annes_sous_responsable_actuel = Column(Integer, nullable=False)     # Temps avec manager actuel
    
    # Variables de formation
    nombre_participation_pee = Column(Integer, nullable=False)  # Participations plan épargne (0-3)
    nb_formations_suivies = Column(Integer, nullable=False)     # Nombre de formations (0-6)
    
    # Variables géographiques
    distance_domicile_travail = Column(Integer, nullable=False) # Distance en km
    
    # Variables d'éducation
    niveau_education = Column(Integer, nullable=False)          # Niveau études (1-5)
    domaine_etude = Column(String(50), nullable=False)          # "Marketing", "Informatique", etc.
    
    # Fréquence de déplacement
    frequence_deplacement = Column(String(20), nullable=False)  # "Occasionnel", etc.
    
    # Variable cible du modèle ML
    a_quitte_l_entreprise = Column(String(5), nullable=False)   # "Oui" ou "Non"
    
    # Métadonnées
    created_at = Column(DateTime, default=func.current_timestamp()) # Date création enregistrement
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())  # Date dernière modification
    
    # Relation vers les requêtes de prédiction (1:N)
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
            "augementation_salaire_precedente": float(self.augementation_salaire_precedente), # Conversion Decimal->float
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
    """
    Sessions de prédiction (single/batch)
    Groupe logique de requêtes ML pour traçabilité
    """
    __tablename__ = 'prediction_sessions'
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_type = Column(String(10), nullable=False)
    total_predictions = Column(Integer, default=0)      # Nombre de prédictions dans la session
    status = Column(String(20), nullable=False, default='pending')  # "pending", "completed", "failed"
    started_at = Column(DateTime, default=func.current_timestamp())
    completed_at = Column(DateTime)
    error_message = Column(Text)
    session_metadata = Column(JSONB)    # Métadonnées JSON libres
    
    # Relations
    prediction_requests = relationship("PredictionRequest", back_populates="session", cascade="all, delete-orphan")   # 1:N vers requêtes
    audit_logs = relationship("APIAuditLog", back_populates="session")  # 1:N vers requêtes
    
    def to_dict(self):
        """Convertit la session en dictionnaire pour l'API (Sérialisation JSON pour API)"""
        return {
            "session_id": str(self.session_id),   # UUID -> string
            "session_type": self.session_type,
            "total_predictions": self.total_predictions,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "session_metadata": self.session_metadata
        }

class PredictionRequest(Base):
    """
    Inputs du modèle ML - Traçabilité des données d'entrée
    Chaque employé prédit génère une requête
    """
    __tablename__ = 'prediction_requests'
    
    request_id = Column(Integer, primary_key=True, autoincrement=True)    # ID unique auto-incrémenté
    session_id = Column(UUID(as_uuid=True), ForeignKey('prediction_sessions.session_id', ondelete='CASCADE'), nullable=False)   # Lien vers session (CASCADE delete)
    employee_id = Column(Integer, ForeignKey('employees.employee_id', ondelete='SET NULL')) # Lien vers employé (optionnel)
    input_data = Column(JSONB, nullable=False)    # Données JSON complètes envoyées au modèle
    request_source = Column(String(20), nullable=False, default='api')  # Source: "api", "batch", "test"
    created_at = Column(DateTime, default=func.current_timestamp()) # Timestamp de création
    
    # Relations bidirectionnelles
    session = relationship("PredictionSession", back_populates="prediction_requests")
    employee = relationship("Employee", back_populates="prediction_requests")
    result = relationship("PredictionResult", back_populates="request", uselist=False, cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convertit la requête en dictionnaire pour l'API"""
        return {
            "request_id": self.request_id,
            "session_id": str(self.session_id),
            "employee_id": self.employee_id,
            "input_data": self.input_data,    # JSONB se sérialise automatiquement
            "request_source": self.request_source,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class PredictionResult(Base):
    """
    Outputs du modèle ML - Traçabilité des résultats de prédiction
    Stocke toutes les informations retournées par le modèle XGBoost
    """
    __tablename__ = 'prediction_results'
    
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(Integer, ForeignKey('prediction_requests.request_id', ondelete='CASCADE'), nullable=False)  # Lien vers requête (CASCADE)
    prediction = Column(String(5), nullable=False)          # "Oui" ou "Non" (attrition)
    probability_quit = Column(Numeric(6,4), nullable=False) # Probabilité de départ (0-1)
    probability_stay = Column(Numeric(6,4), nullable=False) # Probabilité de rester (0-1)
    confidence_level = Column(String(10), nullable=False)   # "Faible", "Moyen", "Élevé"
    risk_factors = Column(ARRAY(Text))  # Liste des facteurs de risque identifiés
    model_version = Column(String(20), nullable=False)      # Version du modèle utilisé
    processing_time_ms = Column(Numeric(10,2))              # Temps de traitement en millisec
    created_at = Column(DateTime, default=func.current_timestamp()) # Timestamp du résultat
    
    # Relations vers la requête
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
    """
    Métadonnées et versioning des modèles ML
    Garde la trace de tous les modèles déployés et leurs performances
    """
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
    """
    Audit complet des appels API
    Traçabilité sécuritaire et monitoring des performances
    """
    __tablename__ = 'api_audit_logs'
    
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('prediction_sessions.session_id', ondelete='SET NULL'))
    endpoint_called = Column(String(100), nullable=False)   # URL endpoint appelé
    http_method = Column(String(10), nullable=False)        # GET, POST, PUT, DELETE
    client_ip = Column(INET)                            # Adresse IP client (type PostgreSQL INET)
    user_agent = Column(Text)                           # User-Agent du navigateur/client
    request_headers = Column(JSONB)                         # Headers HTTP complets
    request_payload = Column(JSONB)                         # Body de la requête JSON
    response_status_code = Column(Integer, nullable=False)  # Code réponse HTTP (200, 404, etc.)
    response_payload = Column(JSONB)                        # Body de la réponse JSON
    response_time_ms = Column(Numeric(10,2))                # Temps de réponse en millisecondes
    created_at = Column(DateTime, default=func.current_timestamp()) # Timestamp de l'appel
    
    # Relation vers session (optionnelle)
    session = relationship("PredictionSession", back_populates="audit_logs")
    
    def to_dict(self):
        """Convertit le log d'audit en dictionnaire pour l'API (Sérialisation JSON pour API)"""
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