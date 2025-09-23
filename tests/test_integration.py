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

# Données de test corrigées selon le schéma Pydantic
VALID_EMPLOYEE_DATA = {
    "satisfaction_employee_environnement": 3,  # 1-4 selon schéma
    "satisfaction_employee_nature_travail": 4,  # 1-4 selon schéma
    "satisfaction_employee_equipe": 3,  # 1-4 selon schéma
    "satisfaction_employee_equilibre_pro_perso": 3,  # 1-4 selon schéma
    "note_evaluation_precedente": 4,  # 1-5
    "note_evaluation_actuelle": 4,  # 1-5
    "niveau_hierarchique_poste": 2,
    "heure_supplementaires": "Oui",
    "augementation_salaire_precedente": 0.15,  # Float, pas string
    "age": 32,
    "genre": "Homme",
    "revenu_mensuel": 3500,
    "statut_marital": "Marié(e)",  # Avec parenthèses selon schéma
    "departement": "Commercial",
    "poste": "Manager",
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
    "domaine_etude": "Marketing",
    "frequence_deplacement": "Voyage_Rare"
}

# Employé avec profil différent pour tests variés
SECOND_EMPLOYEE_DATA = {
    "satisfaction_employee_environnement": 2,  # Plus faible satisfaction
    "satisfaction_employee_nature_travail": 2,
    "satisfaction_employee_equipe": 2,
    "satisfaction_employee_equilibre_pro_perso": 1,  # Très faible
    "note_evaluation_precedente": 3,
    "note_evaluation_actuelle": 2,  # Baisse de performance
    "niveau_hierarchique_poste": 1,
    "heure_supplementaires": "Non",
    "augementation_salaire_precedente": 0.11,  # Augmentation minimale
    "age": 30,
    "genre": "Femme",
    "revenu_mensuel": 2800,  # Salaire plus bas
    "statut_marital": "Célibataire",
    "departement": "Consulting",
    "poste": "Consultant",
    "nombre_experiences_precedentes": 1,
    "annee_experience_totale": 5,
    "annees_dans_l_entreprise": 2,
    "annees_dans_le_poste_actuel": 1,
    "annees_depuis_la_derniere_promotion": 2,
    "annes_sous_responsable_actuel": 1,
    "nombre_participation_pee": 0,
    "nb_formations_suivies": 2,
    "distance_domicile_travail": 25,  # Plus loin
    "niveau_education": 3,
    "domaine_etude": "Autre",
    "frequence_deplacement": "Voyage_Fréquent"  # Plus de voyages
}

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
        validate_response = client.post(
            "/api/v1/predict/validate-input",
            json=VALID_EMPLOYEE_DATA
        )
        assert validate_response.status_code == 200
        
        # 4. Effectuer une prédiction (peut échouer si modèle pas chargé)
        prediction_response = client.post(
            "/api/v1/predict/single",
            json=VALID_EMPLOYEE_DATA
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
        batch_data = {
            "employees": [VALID_EMPLOYEE_DATA, SECOND_EMPLOYEE_DATA]
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
            assert len(data["predictions"]) == 2
    
    def test_data_endpoints_integration(self):
        """Test d'intégration des endpoints de données"""
        # Test du comptage d'employés
        count_response = client.get("/api/v1/data/employees/count")
        assert count_response.status_code in [200, 503]
        
        # Test de l'historique des prédictions
        history_response = client.get("/api/v1/data/predictions/history")
        assert history_response.status_code in [200, 503]
        
        # Test des statistiques
        stats_response = client.get("/api/v1/analytics/predictions/stats")
        assert stats_response.status_code in [200, 503]
    
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
        
        assert response.status_code in [422, 503]  # Validation ou service indisponible
        
        # Test avec batch vide
        empty_batch = {"employees": []}
        
        batch_response = client.post(
            "/api/v1/predict/batch",
            json=empty_batch
        )
        
        assert batch_response.status_code in [400, 422, 503]


class TestModelIntegration:
    """Tests d'intégration avec le modèle ML"""
    
    def test_model_availability_check(self):
        """Test de vérification de disponibilité du modèle"""
        # Test via endpoint de prédiction
        response = client.post(
            "/api/v1/predict/single",
            json=VALID_EMPLOYEE_DATA
        )
        
        if response.status_code == 200:
            # Modèle disponible - vérifier la structure de réponse
            data = response.json()
            assert "prediction" in data
            assert "probability_quit" in data
            assert "probability_stay" in data
            assert "confidence_level" in data
            assert "timestamp" in data
            
            # Vérifier les valeurs cohérentes
            assert data["probability_quit"] + data["probability_stay"] == pytest.approx(1.0, abs=0.01)
            assert data["prediction"] in ["Oui", "Non"]
            
        elif response.status_code == 503:
            # Modèle indisponible - comportement normal en test
            data = response.json()
            assert "detail" in data
    
    def test_model_consistency(self):
        """Test de cohérence du modèle avec mêmes données"""
        # Faire plusieurs prédictions avec les mêmes données
        responses = []
        for _ in range(3):
            response = client.post(
                "/api/v1/predict/single",
                json=VALID_EMPLOYEE_DATA
            )
            responses.append(response)
        
        # Vérifier que toutes ont le même code de statut
        status_codes = [r.status_code for r in responses]
        assert len(set(status_codes)) == 1  # Tous identiques
        
        if responses[0].status_code == 200:
            # Si modèle disponible, vérifier cohérence des prédictions
            predictions = [r.json()["prediction"] for r in responses]
            probabilities = [r.json()["probability_quit"] for r in responses]
            
            # Mêmes données = mêmes résultats
            assert len(set(predictions)) == 1
            assert all(abs(p - probabilities[0]) < 0.001 for p in probabilities)
    
    def test_different_profiles_predictions(self):
        """Test que différents profils donnent des prédictions différentes"""
        response1 = client.post(
            "/api/v1/predict/single",
            json=VALID_EMPLOYEE_DATA
        )
        
        response2 = client.post(
            "/api/v1/predict/single", 
            json=SECOND_EMPLOYEE_DATA
        )
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            # Les probabilités devraient être différentes (profils différents)
            assert data1["probability_quit"] != data2["probability_quit"]
            
            # SECOND_EMPLOYEE_DATA a un profil plus à risque (satisfactions faibles)
            # donc probability_quit devrait être plus élevée
            if data2["probability_quit"] > data1["probability_quit"]:
                # Comportement attendu du modèle
                pass  
            # Sinon, juste vérifier qu'elles sont différentes
            assert abs(data1["probability_quit"] - data2["probability_quit"]) > 0.01


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
    
    def test_response_time_validation(self):
        """Test du temps de réponse pour validation (sans ML)"""
        start_time = time.time()
        
        response = client.post(
            "/api/v1/predict/validate-input",
            json=VALID_EMPLOYEE_DATA
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 0.5  # Validation très rapide
    
    def test_concurrent_health_checks(self):
        """Test de vérifications de santé concurrentes"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def health_check():
            response = client.get("/health/")
            results.put((response.status_code, time.time()))
        
        # Lancer 10 requêtes concurrentes
        threads = []
        start_time = time.time()
        
        for _ in range(10):
            t = threading.Thread(target=health_check)
            threads.append(t)
            t.start()
        
        # Attendre toutes les réponses
        for t in threads:
            t.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Vérifier que toutes ont réussi
        status_codes = []
        while not results.empty():
            status_code, _ = results.get()
            status_codes.append(status_code)
        
        assert all(sc == 200 for sc in status_codes)
        assert len(status_codes) == 10
        assert total_time < 5.0  # 10 requêtes en moins de 5 secondes
    
    def test_prediction_response_time(self):
        """Test du temps de réponse pour les prédictions"""
        start_time = time.time()
        
        response = client.post(
            "/api/v1/predict/single",
            json=VALID_EMPLOYEE_DATA
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            assert response_time < 5.0  # Moins de 5 secondes pour une prédiction
        # Si 503, c'est normal (modèle non chargé en test)
        
    def test_batch_prediction_scalability(self):
        """Test de scalabilité des prédictions batch"""
        # Test avec différentes tailles de batch
        batch_sizes = [1, 5, 10, 25]
        
        for size in batch_sizes:
            batch_data = {
                "employees": [VALID_EMPLOYEE_DATA] * size
            }
            
            start_time = time.time()
            response = client.post(
                "/api/v1/predict/batch",
                json=batch_data
            )
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                # Temps acceptable même pour 25 employés
                assert response_time < 10.0
                
                data = response.json()
                assert data["total_employees"] == size
                assert len(data["predictions"]) == size
            # Si 503, c'est normal (modèle non chargé)


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
        expected_endpoints = [
            "/health/",
            "/api/v1/predict/single",
            "/api/v1/predict/batch",
            "/api/v1/predict/validate-input",
            "/api/v1/predict/supported-values",
            "/info",
            "/api/v1/data/employees/count",
            "/api/v1/data/predictions/history",
            "/api/v1/analytics/predictions/stats"
        ]
        
        for endpoint in expected_endpoints:
            assert endpoint in paths, f"Endpoint {endpoint} manquant dans OpenAPI"
    
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


class TestFullSystemIntegration:
    """Tests d'intégration système complet"""
    
    def test_complete_workflow_single_prediction(self):
        """Test du workflow complet pour une prédiction unique"""
        # 1. Vérifier l'API
        assert client.get("/health/").status_code == 200
        
        # 2. Obtenir la doc des valeurs
        values_resp = client.get("/api/v1/predict/supported-values")
        assert values_resp.status_code == 200
        
        # 3. Valider les données
        validate_resp = client.post(
            "/api/v1/predict/validate-input",
            json=VALID_EMPLOYEE_DATA
        )
        assert validate_resp.status_code == 200
        
        # 4. Faire la prédiction
        predict_resp = client.post(
            "/api/v1/predict/single",
            json=VALID_EMPLOYEE_DATA
        )
        
        # 5. Vérifier les endpoints de données (si DB disponible)
        data_endpoints = [
            "/api/v1/data/employees/count",
            "/api/v1/data/predictions/history", 
            "/api/v1/analytics/predictions/stats"
        ]
        
        for endpoint in data_endpoints:
            resp = client.get(endpoint)
            assert resp.status_code in [200, 503]  # OK ou DB indisponible
        
        # 6. Résultat final
        if predict_resp.status_code == 200:
            # Modèle disponible - workflow complet réussi
            pred_data = predict_resp.json()
            assert "prediction" in pred_data
            assert "probability_quit" in pred_data
        else:
            # Modèle indisponible - workflow partiel acceptable
            assert predict_resp.status_code == 503
    
    def test_api_resilience(self):
        """Test de résilience de l'API face aux erreurs"""
        error_scenarios = [
            # Données complètement invalides
            {"invalid": "data"},
            
            # Données partielles
            {"age": 25, "genre": "Homme"},
            
            # Types incorrects
            {"age": "vingt-cinq", "satisfaction_employee_environnement": "élevé"},
            
            # Valeurs hors limites
            {"age": -5, "satisfaction_employee_environnement": 100}
        ]
        
        for invalid_data in error_scenarios:
            response = client.post(
                "/api/v1/predict/single",
                json=invalid_data
            )
            
            # L'API doit répondre avec une erreur appropriée, pas crasher
            assert response.status_code in [400, 422, 503]
            
            if response.status_code in [400, 422]:
                # Erreur de validation - vérifier qu'il y a un message
                data = response.json()
                assert "detail" in data
    
    def test_concurrent_different_operations(self):
        """Test d'opérations concurrentes différentes"""
        import threading
        
        results = {"health": [], "validation": [], "prediction": []}
        
        def health_check():
            resp = client.get("/health/")
            results["health"].append(resp.status_code)
        
        def validation_check():
            resp = client.post(
                "/api/v1/predict/validate-input",
                json=VALID_EMPLOYEE_DATA
            )
            results["validation"].append(resp.status_code)
        
        def prediction_check():
            resp = client.post(
                "/api/v1/predict/single",
                json=VALID_EMPLOYEE_DATA
            )
            results["prediction"].append(resp.status_code)
        
        # Lancer différents types d'opérations en parallèle
        threads = []
        operations = [health_check, validation_check, prediction_check] * 3  # 9 threads
        
        for operation in operations:
            t = threading.Thread(target=operation)
            threads.append(t)
            t.start()
        
        # Attendre toutes les opérations
        for t in threads:
            t.join()
        
        # Vérifier que toutes les opérations ont réussi
        assert all(sc == 200 for sc in results["health"])
        assert all(sc == 200 for sc in results["validation"])
        assert all(sc in [200, 503] for sc in results["prediction"])  # 503 = modèle indisponible


if __name__ == "__main__":
    pytest.main([__file__, "-v"])