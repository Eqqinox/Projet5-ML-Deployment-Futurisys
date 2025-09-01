"""
Tests d'intégration pour l'API FastAPI
Tests de bout en bout et d'intégration avec le modèle ML
"""

import pytest
from fastapi.testclient import TestClient
import json
import time
from unittest.mock import patch, Mock

from app.main import app
from app.models.ml_model import MLModel

client = TestClient(app)

class TestAPIIntegration:
    """Tests d'intégration de l'API complète"""
    
    def test_api_startup(self):
        """Test que l'API démarre correctement"""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Futurisys ML API"
    
    def test_full_prediction_pipeline(self):
        """Test du pipeline complet de prédiction"""
        # 1. Vérifier la santé de l'API
        health_response = client.get("/health/")
        assert health_response.status_code == 200
        
        # 2. Obtenir les informations sur les valeurs supportées
        values_response = client.get("/api/v1/predict/supported-values")
        assert values_response.status_code == 200
        
        # 3. Valider des données d'entrée
        sample_data = {
            "satisfaction_employee_environnement": 7,
            "satisfaction_employee_nature_travail": 8,
            "satisfaction_employee_equipe": 6,
            "satisfaction_employee_equilibre_pro_perso": 7,
            "note_evaluation_precedente": 4,
            "note_evaluation_actuelle": 4,
            "niveau_hierarchique_poste": 2,
            "heure_supplementaires": "Oui",
            "augementation_salaire_precedente": "Non",
            "age": 32,
            "genre": "Homme",
            "revenu_mensuel": 3500,
            "statut_marital": "Marié",
            "departement": "Recherche et Développement",
            "poste": "Développeur",
            "nombre_experiences_precedentes": 2,
            "annee_experience_totale": 8,
            "annees_dans_l_entreprise": 3,
            "annees_dans_le_poste_actuel": 2,
            "annees_depuis_la_derniere_promotion": 1,
            "annes_sous_responsable_actuel": 2,
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 3,
            "distance_domicile_travail": 15,
            "niveau_education": 4,
            "domaine_etude": "Informatique",
            "frequence_deplacement": "Voyage_Rare"
        }
        
        validate_response = client.post(
            "/api/v1/predict/validate-input",
            json=sample_data
        )
        assert validate_response.status_code == 200
        
        # 4. Effectuer une prédiction (peut échouer si modèle pas chargé)
        prediction_response = client.post(
            "/api/v1/predict/single",
            json=sample_data
        )
        
        # Accepter soit succès (200) soit modèle non disponible (503)
        assert prediction_response.status_code in [200, 503]
        
        if prediction_response.status_code == 200:
            pred_data = prediction_response.json()
            assert "prediction" in pred_data
            assert pred_data["prediction"] in ["Oui", "Non"]
            assert "probability_quit" in pred_data
            assert 0 <= pred_data["probability_quit"] <= 1
    
    def test_batch_prediction_workflow(self):
        """Test du workflow de prédiction batch"""
        sample_employee = {
            "satisfaction_employee_environnement": 5,
            "satisfaction_employee_nature_travail": 6,
            "satisfaction_employee_equipe": 5,
            "satisfaction_employee_equilibre_pro_perso": 4,
            "note_evaluation_precedente": 3,
            "note_evaluation_actuelle": 3,
            "niveau_hierarchique_poste": 2,
            "heure_supplementaires": "Non",
            "augementation_salaire_precedente": "Non",
            "age": 30,
            "genre": "Homme",
            "revenu_mensuel": 3000,
            "statut_marital": "Célibataire",
            "departement": "Recherche et Développement",
            "poste": "Développeur",
            "nombre_experiences_precedentes": 1,
            "annee_experience_totale": 5,
            "annees_dans_l_entreprise": 2,
            "annees_dans_le_poste_actuel": 1,
            "annees_depuis_la_derniere_promotion": 2,
            "annes_sous_responsable_actuel": 1,
            "nombre_participation_pee": 0,
            "nb_formations_suivies": 2,
            "distance_domicile_travail": 10,
            "niveau_education": 3,
            "domaine_etude": "Informatique",
            "frequence_deplacement": "Voyage_Rare"
        }
        
        batch_data = {
            "employees": [sample_employee, sample_employee]
        }
        
        response = client.post(
            "/api/v1/predict/batch",
            json=batch_data
        )
        
        # Accepter soit succès (200) soit modèle non disponible (503)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["total_employees"] == 2
            assert "predictions" in data
            assert "average_quit_probability" in data
    
    def test_error_handling_integration(self):
        """Test de la gestion d'erreur intégrée"""
        # Test avec données invalides
        invalid_data = {
            "age": "pas_un_nombre",
            "satisfaction_employee_environnement": 15  # Hors limites
        }
        
        response = client.post(
            "/api/v1/predict/single",
            json=invalid_data
        )
        
        assert response.status_code == 422
        
        # Test avec batch vide
        empty_batch = {"employees": []}
        
        batch_response = client.post(
            "/api/v1/predict/batch",
            json=empty_batch
        )
        
        assert batch_response.status_code == 422

class TestModelIntegration:
    """Tests d'intégration avec le modèle ML"""
    
    @patch('app.models.ml_model.MLModel.load_model')
    @patch('app.models.ml_model.MLModel.predict_single')
    def test_model_integration_success(self, mock_predict, mock_load):
        """Test d'intégration réussie avec le modèle"""
        # Mock du chargement du modèle
        mock_load.return_value = True
        
        # Mock de la prédiction
        mock_result = Mock()
        mock_result.dict.return_value = {
            "employee_id": None,
            "prediction": "Non",
            "probability_quit": 0.3,
            "probability_stay": 0.7,
            "confidence_level": "Élevé",
            "risk_factors": ["Satisfaction faible"],
            "model_version": "1.0.0",
            "timestamp": "2024-01-01T12:00:00"
        }
        mock_predict.return_value = mock_result
        
        # Test de prédiction
        sample_data = {
            "satisfaction_employee_environnement": 7,
            "satisfaction_employee_nature_travail": 8,
            "satisfaction_employee_equipe": 6,
            "satisfaction_employee_equilibre_pro_perso": 7,
            "note_evaluation_precedente": 4,
            "note_evaluation_actuelle": 4,
            "niveau_hierarchique_poste": 2,
            "heure_supplementaires": "Oui",
            "augementation_salaire_precedente": "Non",
            "age": 32,
            "genre": "Homme",
            "revenu_mensuel": 3500,
            "statut_marital": "Marié",
            "departement": "Recherche et Développement",
            "poste": "Développeur",
            "nombre_experiences_precedentes": 2,
            "annee_experience_totale": 8,
            "annees_dans_l_entreprise": 3,
            "annees_dans_le_poste_actuel": 2,
            "annees_depuis_la_derniere_promotion": 1,
            "annes_sous_responsable_actuel": 2,
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 3,
            "distance_domicile_travail": 15,
            "niveau_education": 4,
            "domaine_etude": "Informatique",
            "frequence_deplacement": "Voyage_Rare"
        }
        
        response = client.post(
            "/api/v1/predict/single",
            json=sample_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Non"
        assert data["probability_quit"] == 0.3

class TestPerformanceIntegration:
    """Tests de performance et de charge"""
    
    def test_response_time_health_check(self):
        """Test du temps de réponse pour les vérifications de santé"""
        start_time = time.time()
        
        response = client.get("/health/")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Moins d'1 seconde
    
    def test_concurrent_health_checks(self):
        """Test de vérifications de santé concurrentes"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def health_check():
            response = client.get("/health/")
            results.put(response.status_code)
        
        # Lancer 5 requêtes concurrentes
        threads = []
        for _ in range(5):
            t = threading.Thread(target=health_check)
            threads.append(t)
            t.start()
        
        # Attendre toutes les réponses
        for t in threads:
            t.join()
        
        # Vérifier que toutes ont réussi
        while not results.empty():
            status_code = results.get()
            assert status_code == 200
    
    @pytest.mark.skipif(
        not hasattr(app.state, 'ml_model') or app.state.ml_model is None,
        reason="Modèle ML non disponible"
    )
    def test_prediction_response_time(self):
        """Test du temps de réponse pour les prédictions (si modèle disponible)"""
        sample_data = {
            "satisfaction_employee_environnement": 7,
            "satisfaction_employee_nature_travail": 8,
            "satisfaction_employee_equipe": 6,
            "satisfaction_employee_equilibre_pro_perso": 7,
            "note_evaluation_precedente": 4,
            "note_evaluation_actuelle": 4,
            "niveau_hierarchique_poste": 2,
            "heure_supplementaires": "Oui",
            "augementation_salaire_precedente": "Non",
            "age": 32,
            "genre": "Homme",
            "revenu_mensuel": 3500,
            "statut_marital": "Marié",
            "departement": "Recherche et Développement",
            "poste": "Développeur",
            "nombre_experiences_precedentes": 2,
            "annee_experience_totale": 8,
            "annees_dans_l_entreprise": 3,
            "annees_dans_le_poste_actuel": 2,
            "annees_depuis_la_derniere_promotion": 1,
            "annes_sous_responsable_actuel": 2,
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 3,
            "distance_domicile_travail": 15,
            "niveau_education": 4,
            "domaine_etude": "Informatique",
            "frequence_deplacement": "Voyage_Rare"
        }
        
        start_time = time.time()
        
        response = client.post(
            "/api/v1/predict/single",
            json=sample_data
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            assert response_time < 5.0  # Moins de 5 secondes pour une prédiction

class TestAPISpecification:
    """Tests de conformité à la spécification OpenAPI"""
    
    def test_openapi_specification(self):
        """Test que la spécification OpenAPI est valide"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        spec = response.json()
        
        # Vérifications basiques du schéma OpenAPI
        assert "openapi" in spec
        assert "info" in spec
        assert spec["info"]["title"] == "Futurisys ML API - Prédiction d'Attrition"
        assert spec["info"]["version"] == "1.0.0"
        assert "paths" in spec
        
        # Vérifier que les endpoints principaux sont documentés
        paths = spec["paths"]
        assert "/health/" in paths
        assert "/api/v1/predict/single" in paths
        assert "/api/v1/predict/batch" in paths
    
    def test_swagger_ui_accessible(self):
        """Test que l'interface Swagger est accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_accessible(self):
        """Test que ReDoc est accessible"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")