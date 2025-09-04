-- =====================================================
-- Script de création de la base de données Futurisys ML
-- Projet 5 - Déploiement modèle XGBoost
-- Dataset: 1470 employés, 27 features + target
-- =====================================================

-- Création de la base de données (à exécuter en tant que superuser)
-- CREATE DATABASE futurisys_ml;
-- CREATE USER futurisys_user WITH ENCRYPTED PASSWORD 'secure_password';
-- GRANT ALL PRIVILEGES ON DATABASE futurisys_ml TO futurisys_user;

-- Se connecter à la base futurisys_ml avant d'exécuter la suite

-- =====================================================
-- ACTIVATION DES EXTENSIONS
-- =====================================================

-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Extension pour fonctions de génération UUID aléatoires (PostgreSQL 13+)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- SUPPRESSION DES TABLES EXISTANTES (si elles existent)
-- =====================================================

DROP TABLE IF EXISTS api_audit_logs CASCADE;
DROP TABLE IF EXISTS prediction_results CASCADE;
DROP TABLE IF EXISTS prediction_requests CASCADE;
DROP TABLE IF EXISTS prediction_sessions CASCADE;
DROP TABLE IF EXISTS model_metadata CASCADE;
DROP TABLE IF EXISTS employees CASCADE;

-- =====================================================
-- TABLE EMPLOYEES (Dataset Projet 4)
-- =====================================================

CREATE TABLE employees (
    -- Clé primaire
    employee_id SERIAL PRIMARY KEY,
    
    -- Variables de satisfaction (1-4)
    satisfaction_employee_environnement INTEGER NOT NULL 
        CHECK (satisfaction_employee_environnement BETWEEN 1 AND 4),
    satisfaction_employee_nature_travail INTEGER NOT NULL 
        CHECK (satisfaction_employee_nature_travail BETWEEN 1 AND 4),
    satisfaction_employee_equipe INTEGER NOT NULL 
        CHECK (satisfaction_employee_equipe BETWEEN 1 AND 4),
    satisfaction_employee_equilibre_pro_perso INTEGER NOT NULL 
        CHECK (satisfaction_employee_equilibre_pro_perso BETWEEN 1 AND 4),
    
    -- Variables d'évaluation (1-4)
    note_evaluation_precedente INTEGER NOT NULL 
        CHECK (note_evaluation_precedente BETWEEN 1 AND 4),
    note_evaluation_actuelle INTEGER NOT NULL 
        CHECK (note_evaluation_actuelle BETWEEN 1 AND 4),
    
    -- Variables hiérarchiques et organisationnelles
    niveau_hierarchique_poste INTEGER NOT NULL 
        CHECK (niveau_hierarchique_poste BETWEEN 1 AND 5),
    
    -- Variables binaires
    heure_supplementaires VARCHAR(5) NOT NULL 
        CHECK (heure_supplementaires IN ('Oui', 'Non')),
    
    -- Variable d'augmentation (pourcentage en float)
    augementation_salaire_precedente DECIMAL(6,4) NOT NULL,
    
    -- Variables démographiques
    age INTEGER NOT NULL 
        CHECK (age BETWEEN 18 AND 60),
    genre VARCHAR(5) NOT NULL 
        CHECK (genre IN ('F', 'M')),
    revenu_mensuel INTEGER NOT NULL 
        CHECK (revenu_mensuel BETWEEN 1000 AND 20000),
    statut_marital VARCHAR(20) NOT NULL 
        CHECK (statut_marital IN ('Célibataire', 'Marié(e)', 'Divorcé(e)')),
    
    -- Variables organisationnelles
    departement VARCHAR(30) NOT NULL 
        CHECK (departement IN ('Commercial', 'Consulting', 'Ressources Humaines')),
    poste VARCHAR(50) NOT NULL 
        CHECK (poste IN (
            'Cadre Commercial',
            'Assistant de Direction',
            'Consultant',
            'Tech Lead',
            'Manager',
            'Senior Manager',
            'Représentant Commercial',
            'Directeur Technique',
            'Ressources Humaines'
        )),
    
    -- Variables d'expérience
    nombre_experiences_precedentes INTEGER NOT NULL 
        CHECK (nombre_experiences_precedentes >= 0),
    annee_experience_totale INTEGER NOT NULL 
        CHECK (annee_experience_totale >= 0),
    annees_dans_l_entreprise INTEGER NOT NULL 
        CHECK (annees_dans_l_entreprise >= 0),
    annees_dans_le_poste_actuel INTEGER NOT NULL 
        CHECK (annees_dans_le_poste_actuel >= 0),
    annees_depuis_la_derniere_promotion INTEGER NOT NULL 
        CHECK (annees_depuis_la_derniere_promotion >= 0),
    annes_sous_responsable_actuel INTEGER NOT NULL 
        CHECK (annes_sous_responsable_actuel >= 0),
    
    -- Variables de formation et développement
    nombre_participation_pee INTEGER NOT NULL 
        CHECK (nombre_participation_pee BETWEEN 0 AND 3),
    nb_formations_suivies INTEGER NOT NULL 
        CHECK (nb_formations_suivies BETWEEN 0 AND 6),
    
    -- Variables géographiques et logistiques
    distance_domicile_travail INTEGER NOT NULL 
        CHECK (distance_domicile_travail BETWEEN 1 AND 29),
    
    -- Variables d'éducation
    niveau_education INTEGER NOT NULL 
        CHECK (niveau_education BETWEEN 1 AND 5),
    domaine_etude VARCHAR(50) NOT NULL 
        CHECK (domaine_etude IN (
            'Infra & Cloud',
            'Autre',
            'Transformation Digitale',
            'Marketing',
            'Entrepreunariat',
            'Ressources Humaines'
        )),
    
    -- Fréquence de déplacement
    frequence_deplacement VARCHAR(20) NOT NULL 
        CHECK (frequence_deplacement IN ('Aucun', 'Occasionnel', 'Frequent')),
    
    -- Variable cible
    a_quitte_l_entreprise VARCHAR(5) NOT NULL 
        CHECK (a_quitte_l_entreprise IN ('Oui', 'Non')),
    
    -- Métadonnées
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- TABLE PREDICTION_SESSIONS (Métadonnées des sessions)
-- =====================================================

CREATE TABLE prediction_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_type VARCHAR(10) NOT NULL 
        CHECK (session_type IN ('single', 'batch')),
    total_predictions INTEGER DEFAULT 0 
        CHECK (total_predictions >= 0),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' 
        CHECK (status IN ('pending', 'completed', 'failed')),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    session_metadata JSONB,
    
    -- Contrainte: si completed, doit avoir une date de fin
    CHECK (
        (status = 'completed' AND completed_at IS NOT NULL) OR
        (status != 'completed')
    )
);

-- =====================================================
-- TABLE PREDICTION_REQUESTS (Inputs du modèle)
-- =====================================================

CREATE TABLE prediction_requests (
    request_id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES prediction_sessions(session_id) ON DELETE CASCADE,
    employee_id INTEGER REFERENCES employees(employee_id) ON DELETE SET NULL,
    input_data JSONB NOT NULL,
    request_source VARCHAR(20) NOT NULL DEFAULT 'api' 
        CHECK (request_source IN ('api', 'batch', 'test')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- TABLE PREDICTION_RESULTS (Outputs du modèle)
-- =====================================================

CREATE TABLE prediction_results (
    result_id SERIAL PRIMARY KEY,
    request_id INTEGER NOT NULL REFERENCES prediction_requests(request_id) ON DELETE CASCADE,
    prediction VARCHAR(5) NOT NULL 
        CHECK (prediction IN ('Oui', 'Non')),
    probability_quit DECIMAL(6,4) NOT NULL 
        CHECK (probability_quit BETWEEN 0 AND 1),
    probability_stay DECIMAL(6,4) NOT NULL 
        CHECK (probability_stay BETWEEN 0 AND 1),
    confidence_level VARCHAR(10) NOT NULL 
        CHECK (confidence_level IN ('Faible', 'Moyen', 'Élevé')),
    risk_factors TEXT[],
    model_version VARCHAR(20) NOT NULL,
    processing_time_ms DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Contrainte: les probabilités doivent sommer à 1.0 (avec tolérance)
    CHECK (ABS((probability_quit + probability_stay) - 1.0) < 0.001)
);

-- =====================================================
-- TABLE MODEL_METADATA (Versioning des modèles)
-- =====================================================

CREATE TABLE model_metadata (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL UNIQUE,
    algorithm_type VARCHAR(50) DEFAULT 'XGBoost',
    threshold_value DECIMAL(6,4) DEFAULT 0.5 
        CHECK (threshold_value BETWEEN 0 AND 1),
    performance_metrics JSONB,
    feature_importance JSONB,
    model_file_path VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deprecated_at TIMESTAMP,
    
    -- Contrainte: si déprécié, doit avoir une date
    CHECK (
        (is_active = FALSE AND deprecated_at IS NOT NULL) OR
        (is_active = TRUE)
    )
);

-- =====================================================
-- TABLE API_AUDIT_LOGS (Audit complet API)
-- =====================================================

CREATE TABLE api_audit_logs (
    log_id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES prediction_sessions(session_id) ON DELETE SET NULL,
    endpoint_called VARCHAR(100) NOT NULL,
    http_method VARCHAR(10) NOT NULL 
        CHECK (http_method IN ('GET', 'POST', 'PUT', 'DELETE', 'PATCH')),
    client_ip INET,
    user_agent TEXT,
    request_headers JSONB,
    request_payload JSONB,
    response_status_code INTEGER NOT NULL 
        CHECK (response_status_code BETWEEN 100 AND 599),
    response_payload JSONB,
    response_time_ms DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- CRÉATION DES INDEX DE PERFORMANCE
-- =====================================================

-- Index pour la table employees
CREATE INDEX idx_employees_departement ON employees(departement);
CREATE INDEX idx_employees_poste ON employees(poste);
CREATE INDEX idx_employees_target ON employees(a_quitte_l_entreprise);
CREATE INDEX idx_employees_age_revenu ON employees(age, revenu_mensuel);
CREATE INDEX idx_employees_created_at ON employees(created_at);

-- Index pour les sessions
CREATE INDEX idx_sessions_type_status ON prediction_sessions(session_type, status);
CREATE INDEX idx_sessions_started_at ON prediction_sessions(started_at);

-- Index pour les requêtes
CREATE INDEX idx_requests_session_id ON prediction_requests(session_id);
CREATE INDEX idx_requests_employee_id ON prediction_requests(employee_id);
CREATE INDEX idx_requests_created_at ON prediction_requests(created_at);

-- Index pour les résultats
CREATE INDEX idx_results_request_id ON prediction_results(request_id);
CREATE INDEX idx_results_prediction ON prediction_results(prediction);
CREATE INDEX idx_results_model_version ON prediction_results(model_version);
CREATE INDEX idx_results_created_at ON prediction_results(created_at);

-- Index pour les métadonnées modèle
CREATE INDEX idx_model_version ON model_metadata(version);
CREATE INDEX idx_model_active ON model_metadata(is_active);

-- Index pour l'audit
CREATE INDEX idx_audit_session_id ON api_audit_logs(session_id);
CREATE INDEX idx_audit_endpoint ON api_audit_logs(endpoint_called);
CREATE INDEX idx_audit_status_code ON api_audit_logs(response_status_code);
CREATE INDEX idx_audit_created_at ON api_audit_logs(created_at);

-- =====================================================
-- TRIGGER POUR UPDATED_AT SUR EMPLOYEES
-- =====================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_employees_updated_at 
    BEFORE UPDATE ON employees 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- INSERTION DU MODÈLE XGBOOST ACTUEL
-- =====================================================

INSERT INTO model_metadata (
    model_name,
    version,
    algorithm_type,
    threshold_value,
    performance_metrics,
    feature_importance,
    model_file_path,
    is_active
) VALUES (
    'XGBoost Employee Attrition Classifier',
    '1.0.0',
    'XGBoost',
    0.514,  -- Seuil optimal du Projet 4
    '{
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
    }'::JSONB,
    '{
        "note": "Feature importance sera ajoutée après import des encodeurs"
    }'::JSONB,
    'app/models/trained_model.pkl',
    TRUE
);

-- =====================================================
-- DONNÉES DE TEST SAMPLE (optionnel)
-- =====================================================

-- Exemple de session de test
INSERT INTO prediction_sessions (session_type, status, completed_at) 
VALUES ('single', 'completed', CURRENT_TIMESTAMP);

-- =====================================================
-- VUES UTILITAIRES POUR ANALYSE
-- =====================================================

-- Vue des prédictions avec détails
CREATE VIEW v_predictions_summary AS
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

-- Vue des statistiques par département
CREATE VIEW v_employee_stats_by_dept AS
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

-- =====================================================
-- GRANTS ET PERMISSIONS
-- =====================================================

-- Accorder les permissions à l'utilisateur de l'application
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO futurisys_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO futurisys_user;
GRANT USAGE ON SCHEMA public TO futurisys_user;

-- =====================================================
-- VERIFICATION DE LA CREATION
-- =====================================================

-- Vérifier les tables créées
SELECT 
    schemaname,
    tablename,
    tableowner,
    tablespace,
    hasindexes,
    hasrules,
    hastriggers
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;

-- Vérifier les contraintes
SELECT 
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type,
    cc.check_clause
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.check_constraints cc 
    ON tc.constraint_name = cc.constraint_name
WHERE tc.table_schema = 'public'
ORDER BY tc.table_name, tc.constraint_name;

-- Vérifier les index
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

COMMENT ON DATABASE futurisys_ml IS 'Base de données pour le projet de déploiement ML - Prédiction attrition employés';
COMMENT ON TABLE employees IS 'Dataset original du Projet 4 - 1470 employés avec 27 features + target';
COMMENT ON TABLE prediction_sessions IS 'Métadonnées des sessions de prédiction (single/batch)';
COMMENT ON TABLE prediction_requests IS 'Inputs envoyés au modèle ML - traçabilité complète';
COMMENT ON TABLE prediction_results IS 'Outputs du modèle ML - résultats des prédictions';
COMMENT ON TABLE model_metadata IS 'Versioning et métadonnées des modèles ML';
COMMENT ON TABLE api_audit_logs IS 'Audit complet de tous les appels API';

-- =====================================================
-- FIN DU SCRIPT
-- =====================================================

-- Ce script crée une base de données complète pour le projet Futurisys ML
-- avec traçabilité complète des prédictions et intégrité des données garantie