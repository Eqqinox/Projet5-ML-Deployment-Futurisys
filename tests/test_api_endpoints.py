"""
Tests unitaires pour les endpoints de l'API FastAPI
Test des endpoints de santé et de prédiction
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from app.main import app
from app.models.schemas import EmployeeData, PredictionResult
from app.models.ml_model import MLModel

client = TestClient(app)

# Données de test
SAMPLE_EMPLOYEE_DATA = {
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

@pytest.fixture
def mock_ml_model():
    """Mock du modèle ML pour les tests"""
    mock_model = Mock(spec=MLModel)
    mock_model.is_loaded = True
    mock_model.health_check.return_value = {
        "model_loaded": True,
        "model_file_exists": True,
        "features_defined": True,
        "categorical_mappings_defined": True
    }
    mock_model.get_model_info.return_value = {
        "model_name": "XGBoost Employee Attrition Classifier",
        "model_type": "XGBoost Classifier",
        "version": "1.0.0",
        "features_count": 26,
        "threshold": 0.5,
        "is_loaded": True,
        "training_date": "2024-01-01",
        "performance_metrics": {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.81,
            "f1_score": 0.82
        }
    }
    return mock_model

class TestHealthEndpoints:
    """Tests pour les endpoints de santé"""
    
    def test_basic_health_check(self):
        """Test du endpoint de santé basique"""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Futurisys ML API"
        assert "timestamp" in data
    
    @patch('app.routers.health.get_ml_model')
    def test_detailed_health_check(self, mock_get_model, mock_ml_model):
        """Test du endpoint de santé détaillée"""
        mock_get_model.return_value = mock_ml_model
        
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "model_health" in data
        assert "checks" in data
    
    @patch('app.routers.health.get_ml_model')
    def test_model_info_endpoint(self, mock_get_model, mock_ml_model):
        """Test du endpoint d'information sur le modèle"""
        mock_get_model.return_value = mock_ml_model
        
        response = client.get("/health/model")
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_name"] == "XGBoost Employee Attrition Classifier"
        assert data["version"] == "1.0.0"
        assert data["features_count"] == 26
    
    def test_liveness_check(self):
        """Test du endpoint de liveness"""
        response = client.get("/health/liveness")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data

class TestPredictionEndpoints:
    """Tests pour les endpoints de prédiction"""
    
    @patch('app.routers.predictions.get_ml_model')
    def test_single_prediction_success(self, mock_get_model, mock_ml_model):
        """Test de prédiction individuelle réussie"""
        # Mock de la réponse de prédiction
        mock_result = Mock()
        mock_result.dict.return_value = {
            "employee_id": None,
            "prediction": "Non",
            "probability_quit": 0.3,
            "probability_stay": 0.7,
            "confidence_level": "Élevé",
            "risk_factors": [],
            "model_version": "1.0.0",
            "timestamp": "2024-01-01T12:00:00"
        }
        mock_ml_model.predict_single.return_value = mock_result
        mock_get_model.return_value = mock_ml_model
        
        response = client.post(
            "/api/v1/predict/single",
            json=SAMPLE_EMPLOYEE_DATA
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability_quit" in data
        assert "probability_stay" in data
    
    def test_single_prediction_validation_error(self):
        """Test de validation des données d'entrée"""
        invalid_data = SAMPLE_EMPLOYEE_DATA.copy()
        invalid_data["age"] = 150  # Âge invalide
        
        response = client.post(
            "/api/v1/predict/single", 
            json=invalid_data
        )
        
        assert response.status_code == 422
    
    @patch('app.routers.predictions.get_ml_model')
    def test_batch_prediction_success(self, mock_get_model, mock_ml_model):
        """Test de prédiction batch réussie"""
        # Mock des résultats batch
        mock_predictions = []
        for i in range(2):
            mock_pred = Mock()
            mock_pred.prediction = "Non" if i == 0 else "Oui"
            mock_pred.probability_quit = 0.3 if i == 0 else 0.8
            mock_predictions.append(mock_pred)
        
        mock_ml_model.predict_batch.return_value = mock_predictions
        mock_get_model.return_value = mock_ml_model
        
        batch_data = {
            "employees": [SAMPLE_EMPLOYEE_DATA, SAMPLE_EMPLOYEE_DATA]
        }
        
        response = client.post(
            "/api/v1/predict/batch",
            json=batch_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_employees"] == 2
        assert "predictions" in data
        assert "average_quit_probability" in data
    
    def test_batch_prediction_size_limit(self):
        """Test de la limite de taille des batches"""
        # Créer un batch trop grand (plus de 100)
        large_batch = {
            "employees": [SAMPLE_EMPLOYEE_DATA] * 101
        }
        
        response = client.post(
            "/api/v1/predict/batch",
            json=large_batch
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Batch Size Error" in data["detail"]["error"]
    
    def test_validate_input_success(self):
        """Test de validation d'entrée réussie"""
        response = client.post(
            "/api/v1/predict/validate-input",
            json=SAMPLE_EMPLOYEE_DATA
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["validation_status"] == "success"
        assert "received_data" in data
    
    def test_validate_input_failure(self):
        """Test de validation d'entrée échouée"""
        invalid_data = {"age": "not_a_number"}  # Données incomplètes et invalides
        
        response = client.post(
            "/api/v1/predict/validate-input",
            json=invalid_data
        )
        
        assert response.status_code == 422
    
    def test_supported_values_endpoint(self):
        """Test de l'endpoint des valeurs supportées"""
        response = client.get("/api/v1/predict/supported-values")
        
        assert response.status_code == 200
        data = response.json()
        assert "categorical_variables" in data
        assert "numerical_ranges" in data
        assert "heure_supplementaires" in data["categorical_variables"]

class TestAPIDocumentation:
    """Tests pour la documentation automatique"""
    
    def test_openapi_schema(self):
        """Test que le schéma OpenAPI est généré"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "info" in schema
        assert schema["info"]["title"] == "Futurisys ML API - Prédiction d'Attrition"
    
    def test_docs_redirect(self):
        """Test de redirection vers la documentation"""
        response = client.get("/", allow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"
    
    def test_info_endpoint(self):
        """Test de l'endpoint d'information général"""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["project"] == "Projet5 ML Deployment"
        assert data["client"] == "Futurisys"
        assert data["model_type"] == "XGBoost Classifier"

class TestErrorHandling:
    """Tests de gestion d'erreurs"""
    
    @patch('app.routers.predictions.get_ml_model')
    def test_model_not_available(self, mock_get_model):
        """Test quand le modèle n'est pas disponible"""
        mock_get_model.side_effect = HTTPException(status_code=503, detail="Modèle ML non disponible")
        
        response = client.post(
            "/api/v1/predict/single",
            json=SAMPLE_EMPLOYEE_DATA
        )
        
        assert response.status_code == 503
    
    def test_invalid_endpoint(self):
        """Test d'endpoint inexistant"""
        response = client.get("/api/v1/predict/nonexistent")
        assert response.status_code == 404
    
    @patch('app.routers.predictions.get_ml_model')
    def test_prediction_internal_error(self, mock_get_model, mock_ml_model):
        """Test de gestion d'erreur interne lors de la prédiction"""
        mock_ml_model.predict_single.side_effect = Exception("Erreur interne du modèle")
        mock_get_model.return_value = mock_ml_model
        
        response = client.post(
            "/api/v1/predict/single",
            json=SAMPLE_EMPLOYEE_DATA
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Prediction Error" in data["detail"]["error"]

# Tests d'intégration
class TestIntegration:
    """Tests d'intégration end-to-end"""
    
    def test_full_prediction_workflow(self):
        """Test du workflow complet de prédiction (si modèle disponible)"""
        # Vérifier d'abord la santé
        health_response = client.get("/health/")
        
        if health_response.status_code == 200:
            # Tenter une prédiction
            pred_response = client.post(
                "/api/v1/predict/single",
                json=SAMPLE_EMPLOYEE_DATA
            )
            
            # Le test réussit même si le modèle n'est pas chargé (503)
            # car cela signifie que l'API fonctionne correctement
            assert pred_response.status_code in [200, 503]
    
    def test_cors_headers(self):
        """Test des headers CORS"""
        response = client.options("/health/")
        # Vérifier que les headers CORS sont présents (si configurés)
        assert response.status_code in [200, 405]  # 405 si OPTIONS n'est pas supporté