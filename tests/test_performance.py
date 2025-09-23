"""
Tests de performance et de charge pour l'API FastAPI
Tests de temps de réponse, concurrence, stress tests et limites système
"""

import pytest
from fastapi.testclient import TestClient
import time
import threading
import queue
import statistics
import concurrent.futures
from unittest.mock import patch

from app.main import app

client = TestClient(app)

# Données de test optimisées pour les performances
PERFORMANCE_TEST_DATA = {
    "satisfaction_employee_environnement": 3,
    "satisfaction_employee_nature_travail": 4,
    "satisfaction_employee_equipe": 3,
    "satisfaction_employee_equilibre_pro_perso": 3,
    "note_evaluation_precedente": 4,
    "note_evaluation_actuelle": 4,
    "niveau_hierarchique_poste": 2,
    "heure_supplementaires": "Oui",
    "augementation_salaire_precedente": 0.15,
    "age": 32,
    "genre": "Homme",
    "revenu_mensuel": 3500,
    "statut_marital": "Marié(e)",
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


class TestResponseTimes:
    """Tests de temps de réponse pour différents endpoints"""
    
    def test_health_endpoint_response_time(self):
        """Test du temps de réponse de l'endpoint de santé"""
        measurements = []
        
        for _ in range(10):
            start_time = time.time()
            response = client.get("/health/")
            end_time = time.time()
            
            assert response.status_code == 200
            measurements.append(end_time - start_time)
        
        avg_time = statistics.mean(measurements)
        max_time = max(measurements)
        
        assert avg_time < 0.1  # Temps moyen < 100ms
        assert max_time < 0.5  # Temps max < 500ms
    
    def test_validation_endpoint_response_time(self):
        """Test du temps de réponse de l'endpoint de validation"""
        measurements = []
        
        for _ in range(10):
            start_time = time.time()
            response = client.post(
                "/api/v1/predict/validate-input",
                json=PERFORMANCE_TEST_DATA
            )
            end_time = time.time()
            
            assert response.status_code == 200
            measurements.append(end_time - start_time)
        
        avg_time = statistics.mean(measurements)
        max_time = max(measurements)
        
        assert avg_time < 0.2  # Temps moyen < 200ms
        assert max_time < 0.8  # Temps max < 800ms
    
    def test_prediction_endpoint_response_time(self):
        """Test du temps de réponse de l'endpoint de prédiction"""
        measurements = []
        
        for _ in range(5):  # Moins d'itérations si modèle lourd
            start_time = time.time()
            response = client.post(
                "/api/v1/predict/single",
                json=PERFORMANCE_TEST_DATA
            )
            end_time = time.time()
            
            if response.status_code == 200:
                measurements.append(end_time - start_time)
            elif response.status_code == 503:
                # Modèle non disponible - test du temps de réponse d'erreur
                measurements.append(end_time - start_time)
                assert end_time - start_time < 0.1  # Erreur rapide
        
        if measurements:
            avg_time = statistics.mean(measurements)
            max_time = max(measurements)
            
            if any(client.post("/api/v1/predict/single", json=PERFORMANCE_TEST_DATA).status_code == 200 for _ in range(1)):
                # Si modèle disponible
                assert avg_time < 3.0  # Temps moyen < 3s
                assert max_time < 5.0  # Temps max < 5s
    
    def test_supported_values_response_time(self):
        """Test du temps de réponse des valeurs supportées"""
        start_time = time.time()
        response = client.get("/api/v1/predict/supported-values")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 0.3  # < 300ms
    
    def test_openapi_spec_response_time(self):
        """Test du temps de génération du schéma OpenAPI"""
        start_time = time.time()
        response = client.get("/openapi.json")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 1.0  # < 1s pour générer le schéma


class TestConcurrency:
    """Tests de concurrence et requêtes simultanées"""
    
    def test_concurrent_health_checks(self):
        """Test de vérifications de santé concurrentes"""
        num_threads = 20
        results = queue.Queue()
        
        def health_check():
            start_time = time.time()
            response = client.get("/health/")
            end_time = time.time()
            results.put({
                'status_code': response.status_code,
                'response_time': end_time - start_time
            })
        
        # Lancer les threads simultanément
        threads = []
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=health_check)
            threads.append(thread)
            thread.start()
        
        # Attendre toutes les réponses
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyser les résultats
        response_times = []
        status_codes = []
        
        while not results.empty():
            result = results.get()
            response_times.append(result['response_time'])
            status_codes.append(result['status_code'])
        
        # Assertions
        assert len(status_codes) == num_threads
        assert all(sc == 200 for sc in status_codes)
        assert total_time < 3.0  # 20 requêtes en moins de 3s
        assert max(response_times) < 1.0  # Aucune requête > 1s
    
    def test_concurrent_validations(self):
        """Test de validations concurrentes"""
        num_requests = 15
        
        def validation_request():
            return client.post(
                "/api/v1/predict/validate-input",
                json=PERFORMANCE_TEST_DATA
            )
        
        # Utiliser ThreadPoolExecutor pour plus de contrôle
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            
            # Soumettre toutes les tâches
            futures = [executor.submit(validation_request) for _ in range(num_requests)]
            
            # Récupérer les résultats
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
        
        # Vérifications
        assert len(responses) == num_requests
        assert all(r.status_code == 200 for r in responses)
        assert total_time < 5.0  # 15 validations en moins de 5s
    
    def test_mixed_concurrent_requests(self):
        """Test de requêtes mixtes concurrentes"""
        results = {'health': [], 'validation': [], 'prediction': [], 'info': []}
        
        def make_request(request_type):
            start_time = time.time()
            
            if request_type == 'health':
                response = client.get("/health/")
            elif request_type == 'validation':
                response = client.post(
                    "/api/v1/predict/validate-input",
                    json=PERFORMANCE_TEST_DATA
                )
            elif request_type == 'prediction':
                response = client.post(
                    "/api/v1/predict/single",
                    json=PERFORMANCE_TEST_DATA
                )
            elif request_type == 'info':
                response = client.get("/info")
            
            end_time = time.time()
            
            results[request_type].append({
                'status_code': response.status_code,
                'response_time': end_time - start_time
            })
        
        # Mélange de différents types de requêtes
        request_types = ['health', 'validation', 'prediction', 'info'] * 5  # 20 requêtes
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            start_time = time.time()
            futures = [executor.submit(make_request, rt) for rt in request_types]
            
            # Attendre toutes les completions
            concurrent.futures.wait(futures)
            total_time = time.time() - start_time
        
        # Vérifications pour chaque type de requête
        assert len(results['health']) == 5
        assert len(results['validation']) == 5
        assert len(results['prediction']) == 5
        assert len(results['info']) == 5
        
        # Tous les health et validation doivent réussir
        assert all(r['status_code'] == 200 for r in results['health'])
        assert all(r['status_code'] == 200 for r in results['validation'])
        assert all(r['status_code'] == 200 for r in results['info'])
        
        # Predictions peuvent être 200 ou 503
        assert all(r['status_code'] in [200, 503] for r in results['prediction'])
        
        assert total_time < 10.0  # 20 requêtes mixtes en moins de 10s


class TestLoadTesting:
    """Tests de charge et stress tests"""
    
    def test_sustained_load_health_endpoint(self):
        """Test de charge soutenue sur l'endpoint de santé"""
        duration_seconds = 10
        request_interval = 0.1  # Une requête toutes les 100ms
        
        results = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            req_start = time.time()
            response = client.get("/health/")
            req_end = time.time()
            
            results.append({
                'status_code': response.status_code,
                'response_time': req_end - req_start,
                'timestamp': req_end - start_time
            })
            
            # Contrôler la fréquence
            time.sleep(max(0, request_interval - (req_end - req_start)))
        
        # Analyse des résultats
        successful_requests = [r for r in results if r['status_code'] == 200]
        avg_response_time = statistics.mean([r['response_time'] for r in successful_requests])
        max_response_time = max([r['response_time'] for r in successful_requests])
        
        assert len(successful_requests) >= duration_seconds * 8  # Au moins 8 req/sec
        assert avg_response_time < 0.2  # Temps moyen < 200ms
        assert max_response_time < 1.0  # Aucune requête > 1s
        assert len(successful_requests) / len(results) > 0.95  # 95% de succès
    
    def test_batch_prediction_performance(self):
        """Test de performance des prédictions par batch"""
        batch_sizes = [1, 5, 10, 25, 50]
        
        for batch_size in batch_sizes:
            batch_data = {
                "employees": [PERFORMANCE_TEST_DATA] * batch_size
            }
            
            start_time = time.time()
            response = client.post(
                "/api/v1/predict/batch",
                json=batch_data
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                # Performance attendue basée sur la taille du batch
                expected_max_time = 0.5 + (batch_size * 0.1)  # 0.5s + 100ms par employé
                assert response_time < expected_max_time
                
                data = response.json()
                assert data["total_employees"] == batch_size
            elif response.status_code == 503:
                # Modèle non disponible - vérifier temps de réponse d'erreur
                assert response_time < 0.2
    
    def test_memory_stability_repeated_requests(self):
        """Test de stabilité mémoire avec requêtes répétées"""
        num_iterations = 100
        response_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            response = client.post(
                "/api/v1/predict/validate-input",
                json=PERFORMANCE_TEST_DATA
            )
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
            
            # Vérifier qu'il n'y a pas de dégradation progressive
            if i > 10:  # Après les premières requêtes
                recent_avg = statistics.mean(response_times[-10:])
                initial_avg = statistics.mean(response_times[:10])
                
                # Le temps moyen ne doit pas augmenter significativement
                assert recent_avg < initial_avg * 3  # Max 3x plus lent
    
    def test_error_handling_performance(self):
        """Test de performance de la gestion d'erreurs"""
        invalid_data_scenarios = [
            {"age": "invalid"},
            {"satisfaction_employee_environnement": 100},
            {},  # Données vides
            {"wrong_field": "value"}
        ]
        
        for invalid_data in invalid_data_scenarios:
            start_time = time.time()
            response = client.post(
                "/api/v1/predict/single",
                json=invalid_data
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Les erreurs doivent être traitées rapidement
            assert response_time < 0.5  # < 500ms pour traiter une erreur
            assert response.status_code in [400, 422, 503]


class TestResourceLimits:
    """Tests des limites de ressources et comportement aux limites"""
    
    def test_large_batch_size_limits(self):
        """Test du comportement avec des batches de taille limite"""
        # Test avec la taille maximale autorisée (100)
        max_batch = {
            "employees": [PERFORMANCE_TEST_DATA] * 100
        }
        
        start_time = time.time()
        response = client.post(
            "/api/v1/predict/batch",
            json=max_batch
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        if response.status_code == 200:
            assert response_time < 30.0  # Max 30s pour 100 employés
            data = response.json()
            assert data["total_employees"] == 100
        elif response.status_code == 503:
            # Modèle non disponible
            assert response_time < 0.5
        
        # Test avec taille excessive (101 - doit être rejeté)
        oversized_batch = {
            "employees": [PERFORMANCE_TEST_DATA] * 101
        }
        
        start_time = time.time()
        response = client.post(
            "/api/v1/predict/batch",
            json=oversized_batch
        )
        end_time = time.time()
        
        # Doit être rejeté rapidement
        assert end_time - start_time < 1.0
        assert response.status_code in [400, 422, 503]
    
    def test_maximum_concurrent_connections(self):
        """Test du nombre maximum de connexions concurrentes"""
        num_connections = 50  # Test avec 50 connexions simultanées
        
        def make_request():
            return client.get("/health/")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_connections) as executor:
            start_time = time.time()
            
            # Lancer toutes les requêtes simultanément
            futures = [executor.submit(make_request) for _ in range(num_connections)]
            
            # Attendre toutes les réponses
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
        
        # Vérifications
        assert len(responses) == num_connections
        successful_responses = [r for r in responses if r.status_code == 200]
        
        # Au moins 90% des requêtes doivent réussir
        success_rate = len(successful_responses) / num_connections
        assert success_rate >= 0.9
        
        # Temps total raisonnable
        assert total_time < 15.0  # 50 requêtes en moins de 15s
    
    def test_data_size_limits(self):
        """Test avec des données de taille limite"""
        # Test avec des chaînes très longues (mais valides)
        large_data = PERFORMANCE_TEST_DATA.copy()
        large_data["domaine_etude"] = "A" * 1000  # Chaîne de 1000 caractères
        
        start_time = time.time()
        response = client.post(
            "/api/v1/predict/validate-input",
            json=large_data
        )
        end_time = time.time()
        
        # Doit être rejeté ou traité rapidement
        assert end_time - start_time < 2.0
        assert response.status_code in [200, 422]  # Valide ou erreur de validation


class TestPerformanceRegression:
    """Tests de régression de performance"""
    
    def test_baseline_performance_metrics(self):
        """Test des métriques de performance de base"""
        # Définir les seuils de performance acceptables
        performance_baselines = {
            'health_check': 0.1,          # 100ms max
            'validation': 0.2,            # 200ms max
            'info_endpoint': 0.15,        # 150ms max
            'openapi_spec': 1.0,          # 1s max
        }
        
        measurements = {}
        
        # Mesurer chaque endpoint
        endpoints_tests = [
            ('health_check', lambda: client.get("/health/")),
            ('validation', lambda: client.post(
                "/api/v1/predict/validate-input", 
                json=PERFORMANCE_TEST_DATA
            )),
            ('info_endpoint', lambda: client.get("/info")),
            ('openapi_spec', lambda: client.get("/openapi.json")),
        ]
        
        for name, test_func in endpoints_tests:
            times = []
            for _ in range(5):  # 5 mesures par endpoint
                start_time = time.time()
                response = test_func()
                end_time = time.time()
                
                assert response.status_code == 200
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            measurements[name] = avg_time
            
            # Vérifier contre la baseline
            assert avg_time < performance_baselines[name], \
                f"{name} trop lent: {avg_time:.3f}s > {performance_baselines[name]}s"
    
    def test_performance_consistency(self):
        """Test de cohérence des performances"""
        num_samples = 20
        measurements = []
        
        for _ in range(num_samples):
            start_time = time.time()
            response = client.get("/health/")
            end_time = time.time()
            
            assert response.status_code == 200
            measurements.append(end_time - start_time)
        
        # Analyser la variabilité
        avg_time = statistics.mean(measurements)
        std_dev = statistics.stdev(measurements)
        
        # La variation ne doit pas être trop importante
        coefficient_of_variation = std_dev / avg_time
        assert coefficient_of_variation < 0.5  # CV < 50%
        
        # Pas d'outliers extrêmes
        max_acceptable = avg_time + (3 * std_dev)
        assert all(t < max_acceptable for t in measurements)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])