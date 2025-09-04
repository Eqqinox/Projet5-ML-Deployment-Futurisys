#!/usr/bin/env python3
"""
Script de création de la base de données Futurisys ML avec SQLAlchemy - VERSION CORRIGÉE
Alternative Python au script SQL
Projet 5 - Déploiement modèle XGBoost
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import json
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, DateTime, 
    Numeric, Text, ForeignKey, Index, text, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET, ARRAY
import uuid

# Configuration simple sans import app
class Settings:
    DATABASE_URL = os.getenv('DATABASE_URL')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_DB = os.getenv('POSTGRES_DB')

settings = Settings()

# Base SQLAlchemy
Base = declarative_base()

# =====================================================
# MODÈLES SQLALCHEMY - VERSION SIMPLIFIÉE
# =====================================================

class Employee(Base):
    """
    Modèle pour la table employees - Dataset Projet 4
    1470 employés avec 27 features + variable cible
    """
    __tablename__ = 'employees'
    
    # Clé primaire
    employee_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Variables de satisfaction (1-4) - Contraintes ajoutées par SQL après
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

# =====================================================
# INDEX DE PERFORMANCE
# =====================================================

# Index pour employees
Index('idx_employees_departement', Employee.departement)
Index('idx_employees_poste', Employee.poste)
Index('idx_employees_target', Employee.a_quitte_l_entreprise)
Index('idx_employees_age_revenu', Employee.age, Employee.revenu_mensuel)
Index('idx_employees_created_at', Employee.created_at)

# Index pour sessions
Index('idx_sessions_type_status', PredictionSession.session_type, PredictionSession.status)
Index('idx_sessions_started_at', PredictionSession.started_at)

# Index pour requêtes
Index('idx_requests_session_id', PredictionRequest.session_id)
Index('idx_requests_employee_id', PredictionRequest.employee_id)
Index('idx_requests_created_at', PredictionRequest.created_at)

# Index pour résultats
Index('idx_results_request_id', PredictionResult.request_id)
Index('idx_results_prediction', PredictionResult.prediction)
Index('idx_results_model_version', PredictionResult.model_version)
Index('idx_results_created_at', PredictionResult.created_at)

# Index pour métadonnées modèle
Index('idx_model_version', ModelMetadata.version)
Index('idx_model_active', ModelMetadata.is_active)

# Index pour audit
Index('idx_audit_session_id', APIAuditLog.session_id)
Index('idx_audit_endpoint', APIAuditLog.endpoint_called)
Index('idx_audit_status_code', APIAuditLog.response_status_code)
Index('idx_audit_created_at', APIAuditLog.created_at)

# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def create_database_url(host: str = None, port: int = None, user: str = None, 
                       password: str = None, database: str = None) -> str:
    """Crée l'URL de connexion à la base de données"""
    host = host or settings.POSTGRES_HOST
    port = port or settings.POSTGRES_PORT
    user = user or settings.POSTGRES_USER
    password = password or settings.POSTGRES_PASSWORD
    database = database or settings.POSTGRES_DB
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

def create_engine_with_settings(database_url: str = None, echo: bool = False):
    """Crée le moteur SQLAlchemy avec les paramètres optimaux"""
    if not database_url:
        database_url = settings.DATABASE_URL or create_database_url()
    
    return create_engine(
        database_url,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True
    )

def verify_database_connection(engine):
    """Vérifie la connexion à la base de données"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Connexion PostgreSQL réussie: {version}")
            return True
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False

def create_extensions(engine):
    """Crée les extensions nécessaires"""
    extensions = [
        'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";',
        'CREATE EXTENSION IF NOT EXISTS "pgcrypto";'
    ]
    
    try:
        with engine.connect() as conn:
            for ext in extensions:
                conn.execute(text(ext))
                conn.commit()
        print("✅ Extensions créées avec succès")
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des extensions: {e}")

def add_check_constraints(engine):
    """Ajoute les contraintes CHECK après création des tables"""
    constraints = [
        # Contraintes pour employees
        "ALTER TABLE employees ADD CONSTRAINT check_satisfaction_env CHECK (satisfaction_employee_environnement BETWEEN 1 AND 4);",
        "ALTER TABLE employees ADD CONSTRAINT check_satisfaction_travail CHECK (satisfaction_employee_nature_travail BETWEEN 1 AND 4);",
        "ALTER TABLE employees ADD CONSTRAINT check_satisfaction_equipe CHECK (satisfaction_employee_equipe BETWEEN 1 AND 4);",
        "ALTER TABLE employees ADD CONSTRAINT check_satisfaction_equilibre CHECK (satisfaction_employee_equilibre_pro_perso BETWEEN 1 AND 4);",
        "ALTER TABLE employees ADD CONSTRAINT check_note_eval_prec CHECK (note_evaluation_precedente BETWEEN 1 AND 4);",
        "ALTER TABLE employees ADD CONSTRAINT check_note_eval_act CHECK (note_evaluation_actuelle BETWEEN 1 AND 4);",
        "ALTER TABLE employees ADD CONSTRAINT check_niveau_hier CHECK (niveau_hierarchique_poste BETWEEN 1 AND 5);",
        "ALTER TABLE employees ADD CONSTRAINT check_heures_sup CHECK (heure_supplementaires IN ('Oui', 'Non'));",
        "ALTER TABLE employees ADD CONSTRAINT check_age CHECK (age BETWEEN 18 AND 60);",
        "ALTER TABLE employees ADD CONSTRAINT check_genre CHECK (genre IN ('F', 'M'));",
        "ALTER TABLE employees ADD CONSTRAINT check_revenu CHECK (revenu_mensuel BETWEEN 1000 AND 20000);",
        "ALTER TABLE employees ADD CONSTRAINT check_statut_marital CHECK (statut_marital IN ('Célibataire', 'Marié(e)', 'Divorcé(e)'));",
        "ALTER TABLE employees ADD CONSTRAINT check_departement CHECK (departement IN ('Commercial', 'Consulting', 'Ressources Humaines'));",
        "ALTER TABLE employees ADD CONSTRAINT check_poste CHECK (poste IN ('Cadre Commercial', 'Assistant de Direction', 'Consultant', 'Tech Lead', 'Manager', 'Senior Manager', 'Représentant Commercial', 'Directeur Technique', 'Ressources Humaines'));",
        "ALTER TABLE employees ADD CONSTRAINT check_domaine_etude CHECK (domaine_etude IN ('Infra & Cloud', 'Autre', 'Transformation Digitale', 'Marketing', 'Entrepreunariat', 'Ressources Humaines'));",
        "ALTER TABLE employees ADD CONSTRAINT check_frequence_depl CHECK (frequence_deplacement IN ('Aucun', 'Occasionnel', 'Frequent'));",
        "ALTER TABLE employees ADD CONSTRAINT check_target CHECK (a_quitte_l_entreprise IN ('Oui', 'Non'));",
        "ALTER TABLE employees ADD CONSTRAINT check_exp_prec CHECK (nombre_experiences_precedentes >= 0);",
        "ALTER TABLE employees ADD CONSTRAINT check_exp_totale CHECK (annee_experience_totale >= 0);",
        "ALTER TABLE employees ADD CONSTRAINT check_anciennete_entreprise CHECK (annees_dans_l_entreprise >= 0);",
        "ALTER TABLE employees ADD CONSTRAINT check_anciennete_poste CHECK (annees_dans_le_poste_actuel >= 0);",
        "ALTER TABLE employees ADD CONSTRAINT check_promo CHECK (annees_depuis_la_derniere_promotion >= 0);",
        "ALTER TABLE employees ADD CONSTRAINT check_responsable CHECK (annes_sous_responsable_actuel >= 0);",
        "ALTER TABLE employees ADD CONSTRAINT check_pee CHECK (nombre_participation_pee BETWEEN 0 AND 3);",
        "ALTER TABLE employees ADD CONSTRAINT check_formations CHECK (nb_formations_suivies BETWEEN 0 AND 6);",
        "ALTER TABLE employees ADD CONSTRAINT check_distance CHECK (distance_domicile_travail BETWEEN 1 AND 29);",
        "ALTER TABLE employees ADD CONSTRAINT check_education CHECK (niveau_education BETWEEN 1 AND 5);",
        
        # Contraintes pour autres tables
        "ALTER TABLE prediction_sessions ADD CONSTRAINT check_session_type CHECK (session_type IN ('single', 'batch'));",
        "ALTER TABLE prediction_sessions ADD CONSTRAINT check_session_status CHECK (status IN ('pending', 'completed', 'failed'));",
        "ALTER TABLE prediction_requests ADD CONSTRAINT check_request_source CHECK (request_source IN ('api', 'batch', 'test'));",
        "ALTER TABLE prediction_results ADD CONSTRAINT check_prediction CHECK (prediction IN ('Oui', 'Non'));",
        "ALTER TABLE prediction_results ADD CONSTRAINT check_prob_quit CHECK (probability_quit BETWEEN 0 AND 1);",
        "ALTER TABLE prediction_results ADD CONSTRAINT check_prob_stay CHECK (probability_stay BETWEEN 0 AND 1);",
        "ALTER TABLE prediction_results ADD CONSTRAINT check_confidence CHECK (confidence_level IN ('Faible', 'Moyen', 'Élevé'));",
        "ALTER TABLE api_audit_logs ADD CONSTRAINT check_http_method CHECK (http_method IN ('GET', 'POST', 'PUT', 'DELETE', 'PATCH'));",
        "ALTER TABLE api_audit_logs ADD CONSTRAINT check_status_code CHECK (response_status_code BETWEEN 100 AND 599);"
    ]
    
    try:
        with engine.connect() as conn:
            for constraint in constraints:
                try:
                    conn.execute(text(constraint))
                    conn.commit()
                except Exception as e:
                    if "already exists" not in str(e):
                        print(f"⚠️ Erreur contrainte: {e}")
        print("✅ Contraintes CHECK ajoutées avec succès")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'ajout des contraintes: {e}")

def create_all_tables(engine, drop_existing: bool = False):
    """Crée toutes les tables de la base de données"""
    try:
        if drop_existing:
            print("🗑️ Suppression des tables existantes...")
            Base.metadata.drop_all(engine)
            
        print("🏗️ Création des tables...")
        Base.metadata.create_all(engine)
        print("✅ Tables créées avec succès")
        
        # Ajouter les contraintes CHECK après création des tables
        add_check_constraints(engine)
        
        # Vérification des tables créées
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """))
            tables = [row[0] for row in result.fetchall()]
            print(f"📋 Tables créées: {', '.join(tables)}")
            
    except Exception as e:
        print(f"❌ Erreur lors de la création des tables: {e}")
        raise e

def insert_initial_model_metadata(session):
    """Insère les métadonnées du modèle XGBoost initial"""
    try:
        # Vérifier si le modèle existe déjà
        existing_model = session.query(ModelMetadata).filter_by(version="1.0.0").first()
        if existing_model:
            print("ℹ️ Modèle v1.0.0 déjà présent")
            return existing_model
        
        # Métadonnées du modèle du Projet 4
        performance_metrics = {
            "accuracy": 0.8588,
            "accuracy_std": 0.0220,
            "precision": 0.5654,
            "precision_std": 0.0701,
            "recall": 0.5684,
            "recall_std": 0.0678,
            "f1_score": 0.5656,
            "f1_std": 0.0638,
            "roc_auc": 0.8252,
            "roc_auc_std": 0.0212
        }
        
        model = ModelMetadata(
            model_name="XGBoost Employee Attrition Classifier",
            version="1.0.0",
            algorithm_type="XGBoost",
            threshold_value=0.514,  # Seuil optimal du Projet 4
            performance_metrics=performance_metrics,
            model_file_path="app/models/trained_model.pkl",
            is_active=True
        )
        
        session.add(model)
        session.commit()
        print("✅ Métadonnées du modèle XGBoost ajoutées")
        return model
        
    except Exception as e:
        print(f"❌ Erreur lors de l'insertion des métadonnées: {e}")
        session.rollback()
        raise e

def create_views(engine):
    """Crée les vues utilitaires pour l'analyse"""
    views = [
        """
        CREATE OR REPLACE VIEW v_predictions_summary AS
        SELECT 
            ps.session_id,
            ps.session_type,
            ps.started_at,
            pr.prediction,
            pr.probability_quit,
            pr.confidence_level,
            pr.model_version,
            pr.created_at as prediction_time
        FROM prediction_sessions ps
        JOIN prediction_requests req ON ps.session_id = req.session_id
        JOIN prediction_results pr ON req.request_id = pr.request_id
        ORDER BY pr.created_at DESC;
        """,
        
        """
        CREATE OR REPLACE VIEW v_employee_stats_by_dept AS
        SELECT 
            departement,
            COUNT(*) as total_employees,
            COUNT(CASE WHEN a_quitte_l_entreprise = 'Oui' THEN 1 END) as quit_count,
            ROUND(
                COUNT(CASE WHEN a_quitte_l_entreprise = 'Oui' THEN 1 END) * 100.0 / COUNT(*), 
                2
            ) as quit_percentage,
            AVG(age) as avg_age,
            AVG(revenu_mensuel) as avg_salary
        FROM employees
        GROUP BY departement
        ORDER BY quit_percentage DESC;
        """
    ]
    
    try:
        with engine.connect() as conn:
            for view in views:
                conn.execute(text(view))
                conn.commit()
        print("✅ Vues utilitaires créées")
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des vues: {e}")

def create_database(drop_existing: bool = False, with_sample_data: bool = True, 
                   database_url: str = None, verbose: bool = True):
    """Fonction principale pour créer la base de données complète"""
    print("🚀 Début de la création de la base de données Futurisys ML")
    print("=" * 60)
    
    try:
        # 1. Création du moteur
        engine = create_engine_with_settings(database_url, echo=verbose)
        
        # 2. Vérification de la connexion
        if not verify_database_connection(engine):
            raise Exception("Impossible de se connecter à la base de données")
        
        # 3. Création des extensions
        create_extensions(engine)
        
        # 4. Création des tables
        create_all_tables(engine, drop_existing)
        
        # 5. Création des vues
        create_views(engine)
        
        # 6. Session pour les données initiales
        SessionLocal = sessionmaker(bind=engine)
        db_session = SessionLocal()
        
        try:
            # 7. Insertion du modèle initial
            insert_initial_model_metadata(db_session)
                
        finally:
            db_session.close()
        
        print("=" * 60)
        print("✅ Base de données créée avec succès !")
        print(f"🔗 URL de connexion: {database_url or settings.DATABASE_URL}")
        print(f"📊 Tables créées: {len(Base.metadata.tables)}")
        print("🎯 La base est prête pour l'import du dataset et l'utilisation par l'API")
        
        return engine
        
    except Exception as e:
        print(f"❌ Erreur lors de la création: {e}")
        raise e

# =====================================================
# POINT D'ENTRÉE PRINCIPAL
# =====================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Création de la base de données Futurisys ML")
    parser.add_argument("--drop", action="store_true", help="Supprimer les tables existantes")
    parser.add_argument("--quiet", action="store_true", help="Mode silencieux")
    parser.add_argument("--url", type=str, help="URL de connexion personnalisée")
    
    args = parser.parse_args()
    
    try:
        # Création de la base
        engine = create_database(
            drop_existing=args.drop,
            database_url=args.url,
            verbose=not args.quiet
        )
        
        print("🎉 Script terminé avec succès !")
            
    except Exception as e:
        print(f"💥 Échec de la création: {e}")
        sys.exit(1)