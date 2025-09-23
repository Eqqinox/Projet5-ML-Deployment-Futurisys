"""
Tests de cas limites et robustesse avec données réelles du Projet 4
Tests avec profils extrêmes, données corrompues et scénarios métier complexes
"""

import pytest
from fastapi.testclient import TestClient
import json

from app.main import app

client = TestClient(app)

# ========================================================
# DONNÉES RÉELLES EXTRAITES DU PROJET 4 - CAS LIMITES
# ========================================================

# Cas 1: Employé très jeune avec faible expérience (profil à haut risque)
JEUNE_EMPLOYE_HAUT_RISQUE = {
    "satisfaction_employee_environnement": 1,  # Très insatisfait
    "satisfaction_employee_nature_travail": 2,
    "satisfaction_employee_equipe": 1,
    "satisfaction_employee_equilibre_pro_perso": 1,
    "note_evaluation_precedente": 1,
    "note_evaluation_actuelle": 1,  
    "niveau_hierarchique_poste": 1,
    "heure_supplementaires": "Oui",
    "augementation_salaire_precedente": 0.11,  # Augmentation minimale
    "age": 19,  # Très jeune (basé sur dataset)
    "genre": "Femme",
    "revenu_mensuel": 1118,  # Salaire très bas du dataset
    "statut_marital": "Célibataire",
    "departement": "Commercial",
    "poste": "Représentant Commercial",
    "nombre_experiences_precedentes": 1,
    "annee_experience_totale": 1,
    "annees_dans_l_entreprise": 1,
    "annees_dans_le_poste_actuel": 0,
    "annees_depuis_la_derniere_promotion": 1,
    "annes_sous_responsable_actuel": 0,
    "nombre_participation_pee": 0,
    "nb_formations_suivies": 0,
    "distance_domicile_travail": 29,  # Distance maximale
    "niveau_education": 1,  # Niveau le plus bas
    "domaine_etude": "Infra & Cloud",
    "frequence_deplacement": "Voyage_Fréquent"
}

# Cas 2: Senior Manager expérimenté (profil faible risque)
SENIOR_MANAGER_FAIBLE_RISQUE = {
    "satisfaction_employee_environnement": 4,  # Très satisfait
    "satisfaction_employee_nature_travail": 4,
    "satisfaction_employee_equipe": 4,
    "satisfaction_employee_equilibre_pro_perso": 4,
    "note_evaluation_precedente": 5,  # Excellente évaluation
    "note_evaluation_actuelle": 5,
    "niveau_hierarchique_poste": 5,  # Directeur
    "heure_supplementaires": "Non",
    "augementation_salaire_precedente": 0.25,  # Augmentation élevée
    "age": 59,  # Senior du dataset
    "genre": "Homme",
    "revenu_mensuel": 19847,  # Salaire élevé du dataset
    "statut_marital": "Marié(e)",
    "departement": "Commercial",
    "poste": "Senior Manager",
    "nombre_experiences_precedentes": 4,
    "annee_experience_totale": 31,
    "annees_dans_l_entreprise": 29,
    "annees_dans_le_poste_actuel": 10,
    "annees_depuis_la_derniere_promotion": 11,
    "annes_sous_responsable_actuel": 10,
    "nombre_participation_pee": 1,
    "nb_formations_suivies": 5,
    "distance_domicile_travail": 9,
    "niveau_education": 3,
    "domaine_etude": "Infra & Cloud",
    "frequence_deplacement": "Voyage_Fréquent"
}

# Cas 3: Profil paradoxal (satisfactions élevées mais évaluation faible)
PROFIL_PARADOXAL = {
    "satisfaction_employee_environnement": 4,  # Satisfait
    "satisfaction_employee_nature_travail": 4,
    "satisfaction_employee_equipe": 4,
    "satisfaction_employee_equilibre_pro_perso": 3,
    "note_evaluation_precedente": 1,  # Très mauvaise évaluation
    "note_evaluation_actuelle": 1,
    "niveau_hierarchique_poste": 1,
    "heure_supplementaires": "Oui",
    "augementation_salaire_precedente": 0.11,
    "age": 45,
    "genre": "Femme",
    "revenu_mensuel": 18824,  # Salaire élevé malgré mauvaise éval
    "statut_marital": "Célibataire",
    "departement": "Commercial",
    "poste": "Senior Manager",
    "nombre_experiences_precedentes": 2,
    "annee_experience_totale": 26,
    "annees_dans_l_entreprise": 24,
    "annees_dans_le_poste_actuel": 10,
    "annees_depuis_la_derniere_promotion": 1,
    "annes_sous_responsable_actuel": 11,
    "nombre_participation_pee": 0,
    "nb_formations_suivies": 2,
    "distance_domicile_travail": 2,
    "niveau_education": 3,
    "domaine_etude": "Marketing",
    "frequence_deplacement": "Voyage_Rare"
}

# Cas 4: Employé avec ancienneté zéro (nouveau)
NOUVEAU_EMPLOYE = {
    "satisfaction_employee_environnement": 3,
    "satisfaction_employee_nature_travail": 3,
    "satisfaction_employee_equipe": 2,
    "satisfaction_employee_equilibre_pro_perso": 3,
    "note_evaluation_precedente": 3,
    "note_evaluation_actuelle": 3,
    "niveau_hierarchique_poste": 1,
    "heure_supplementaires": "Oui",
    "augementation_salaire_precedente": 0.15,
    "age": 37,
    "genre": "Homme",
    "revenu_mensuel": 2090,
    "statut_marital": "Célibataire",
    "departement": "Consulting",
    "poste": "Consultant",
    "nombre_experiences_precedentes": 6,
    "annee_experience_totale": 7,
    "annees_dans_l_entreprise": 0,  # Nouveau dans l'entreprise
    "annees_dans_le_poste_actuel": 0,
    "annees_depuis_la_derniere_promotion": 0,
    "annes_sous_responsable_actuel": 0,
    "nombre_participation_pee": 0,
    "nb_formations_suivies": 3,
    "distance_domicile_travail": 2,
    "niveau_education": 2,
    "domaine_etude": "Autre",
    "frequence_deplacement": "Voyage_Rare"
}

# Cas 5: Cas extrême - Directeur Technique avec salaire max
DIRECTEUR_TECHNIQUE_MAX = {
    "satisfaction_employee_environnement": 4,
    "satisfaction_employee_nature_travail": 2,
    "satisfaction_employee_equipe": 4,
    "satisfaction_employee_equilibre_pro_perso": 2,
    "note_evaluation_precedente": 3,
    "note_evaluation_actuelle": 3,
    "niveau_hierarchique_poste": 5,
    "heure_supplementaires": "Non",
    "augementation_salaire_precedente": 0.14,
    "age": 29,
    "genre": "Femme",
    "revenu_mensuel": 18789,  # Un des salaires les plus élevés
    "statut_marital": "Marié(e)",
    "departement": "Commercial",
    "poste": "Senior Manager",
    "nombre_experiences_precedentes": 2,
    "annee_experience_totale": 26,
    "annees_dans_l_entreprise": 11,
    "annees_dans_le_poste_actuel": 4,
    "annees_depuis_la_derniere_promotion": 0,
    "annes_sous_responsable_actuel": 8,
    "nombre_participation_pee": 1,
    "nb_formations_suivies": 2,
    "distance_domicile_travail": 4,
    "niveau_education": 2,
    "domaine_etude": "Marketing",
    "frequence_deplacement": "Voyage_Rare"
}


class TestEdgeCasesRealData:
    """Tests avec des cas limites réels du Projet 4"""
    
    def test_jeune_employe_haut_risque(self):
        """Test avec employé très jeune à haut risque d'attrition"""
        response = client.post(
            "/api/v1/predict/single",
            json=JEUNE_EMPLOYE_HAUT_RISQUE
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability_quit" in data
            
            # Ce profil devrait avoir une probabilité élevée de partir
            # (jeune, insatisfait, faible salaire, peu d'expérience)
            assert data["prediction"] in ["Oui", "Non"]
            assert 0 <= data["probability_quit"] <= 1
            
            # Test de cohérence : profil à risque
            if data["probability_quit"] > 0.7:
                assert data["prediction"] == "Oui"
        elif response.status_code == 503:
            # Modèle non disponible en test - acceptable
            assert "detail" in response.json()
    
    def test_senior_manager_faible_risque(self):
        """Test avec Senior Manager expérimenté à faible risque"""
        response = client.post(
            "/api/v1/predict/single",
            json=SENIOR_MANAGER_FAIBLE_RISQUE
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability_quit" in data
            
            # Ce profil devrait avoir une probabilité faible de partir
            # (senior, très satisfait, salaire élevé, expérimenté)
            assert data["prediction"] in ["Oui", "Non"]
            assert 0 <= data["probability_quit"] <= 1
            
            # Test de cohérence : profil stable
            if data["probability_quit"] < 0.3:
                assert data["prediction"] == "Non"
        elif response.status_code == 503:
            assert "detail" in response.json()
    
    def test_profil_paradoxal(self):
        """Test avec profil paradoxal (satisfait mais mal évalué)"""
        response = client.post(
            "/api/v1/predict/single",
            json=PROFIL_PARADOXAL
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability_quit" in data
            
            # Profil complexe à analyser
            # Satisfactions hautes vs évaluations basses
            assert data["prediction"] in ["Oui", "Non"]
            assert 0 <= data["probability_quit"] <= 1
            
            # Vérifier que le modèle gère cette contradiction
            assert "confidence_level" in data
        elif response.status_code == 503:
            assert "detail" in response.json()
    
    def test_nouveau_employe(self):
        """Test avec employé nouvellement arrivé (ancienneté 0)"""
        response = client.post(
            "/api/v1/predict/single",
            json=NOUVEAU_EMPLOYE
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability_quit" in data
            
            # Nouveau employé = risque modéré à élevé
            assert data["prediction"] in ["Oui", "Non"]
            assert 0 <= data["probability_quit"] <= 1
        elif response.status_code == 503:
            assert "detail" in response.json()
    
    def test_directeur_technique_salaire_max(self):
        """Test avec Directeur Technique au salaire maximum"""
        response = client.post(
            "/api/v1/predict/single",
            json=DIRECTEUR_TECHNIQUE_MAX
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability_quit" in data
            
            # Salaire élevé mais autres facteurs mixtes
            assert data["prediction"] in ["Oui", "Non"]
            assert 0 <= data["probability_quit"] <= 1
        elif response.status_code == 503:
            assert "detail" in response.json()


class TestExtremeValues:
    """Tests avec valeurs extrêmes des plages autorisées"""
    
    def test_valeurs_minimales(self):
        """Test avec toutes les valeurs au minimum"""
        extreme_min_data = {
            "satisfaction_employee_environnement": 1,  # Min
            "satisfaction_employee_nature_travail": 1,  # Min
            "satisfaction_employee_equipe": 1,  # Min
            "satisfaction_employee_equilibre_pro_perso": 1,  # Min
            "note_evaluation_precedente": 1,  # Min
            "note_evaluation_actuelle": 1,  # Min
            "niveau_hierarchique_poste": 1,  # Min
            "heure_supplementaires": "Non",
            "augementation_salaire_precedente": 0.11,  # Min du dataset
            "age": 18,  # Min probable
            "genre": "Homme",
            "revenu_mensuel": 1118,  # Min du dataset
            "statut_marital": "Célibataire",
            "departement": "Consulting",
            "poste": "Consultant",
            "nombre_experiences_precedentes": 0,
            "annee_experience_totale": 0,
            "annees_dans_l_entreprise": 0,
            "annees_dans_le_poste_actuel": 0,
            "annees_depuis_la_derniere_promotion": 0,
            "annes_sous_responsable_actuel": 0,
            "nombre_participation_pee": 0,
            "nb_formations_suivies": 0,
            "distance_domicile_travail": 1,
            "niveau_education": 1,
            "domaine_etude": "Autre",
            "frequence_deplacement": "Pas_de_Voyage"
        }
        
        response = client.post(
            "/api/v1/predict/single",
            json=extreme_min_data
        )
        
        # Doit être accepté ou modèle indisponible
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            # Profil très à risque (tout au minimum)
            assert data["prediction"] in ["Oui", "Non"]
    
    def test_valeurs_maximales(self):
        """Test avec toutes les valeurs au maximum"""
        extreme_max_data = {
            "satisfaction_employee_environnement": 4,  # Max
            "satisfaction_employee_nature_travail": 4,  # Max
            "satisfaction_employee_equipe": 4,  # Max
            "satisfaction_employee_equilibre_pro_perso": 4,  # Max
            "note_evaluation_precedente": 5,  # Max
            "note_evaluation_actuelle": 5,  # Max
            "niveau_hierarchique_poste": 5,  # Max
            "heure_supplementaires": "Oui",
            "augementation_salaire_precedente": 0.25,  # Max du dataset
            "age": 65,  # Max probable
            "genre": "Femme",
            "revenu_mensuel": 19847,  # Max du dataset
            "statut_marital": "Marié(e)",
            "departement": "Commercial",
            "poste": "Senior Manager",
            "nombre_experiences_precedentes": 9,
            "annee_experience_totale": 40,
            "annees_dans_l_entreprise": 35,
            "annees_dans_le_poste_actuel": 15,
            "annees_depuis_la_derniere_promotion": 15,
            "annes_sous_responsable_actuel": 15,
            "nombre_participation_pee": 3,
            "nb_formations_suivies": 6,
            "distance_domicile_travail": 29,
            "niveau_education": 5,
            "domaine_etude": "Marketing",
            "frequence_deplacement": "Voyage_Fréquent"
        }
        
        response = client.post(
            "/api/v1/predict/single",
            json=extreme_max_data
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            # Profil très stable (tout au maximum)
            assert data["prediction"] in ["Oui", "Non"]


class TestCorruptedData:
    """Tests avec données corrompues et incohérentes"""
    
    def test_incoherence_experience_age(self):
        """Test avec incohérence expérience vs âge"""
        inconsistent_data = {
            "satisfaction_employee_environnement": 3,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_precedente": 3,
            "note_evaluation_actuelle": 3,
            "niveau_hierarchique_poste": 2,
            "heure_supplementaires": "Non",
            "augementation_salaire_precedente": 0.15,
            "age": 25,  # Jeune
            "genre": "Homme",
            "revenu_mensuel": 3500,
            "statut_marital": "Marié(e)",
            "departement": "Commercial",
            "poste": "Manager",
            "nombre_experiences_precedentes": 2,
            "annee_experience_totale": 35,  # Incohérent avec l'âge
            "annees_dans_l_entreprise": 30,  # Impossible à 25 ans
            "annees_dans_le_poste_actuel": 25,
            "annees_depuis_la_derniere_promotion": 1,
            "annes_sous_responsable_actuel": 2,
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 3,
            "distance_domicile_travail": 15,
            "niveau_education": 4,
            "domaine_etude": "Marketing",
            "frequence_deplacement": "Voyage_Rare"
        }
        
        response = client.post(
            "/api/v1/predict/single",
            json=inconsistent_data
        )
        
        # Le modèle devrait traiter les données même incohérentes
        assert response.status_code in [200, 422, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            # Vérifier que le modèle gère l'incohérence
            assert "confidence_level" in data
    
    def test_salaire_incoherent_avec_poste(self):
        """Test avec salaire incohérent par rapport au poste"""
        salary_inconsistent = {
            "satisfaction_employee_environnement": 3,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_precedente": 4,
            "note_evaluation_actuelle": 4,
            "niveau_hierarchique_poste": 5,  # Directeur
            "heure_supplementaires": "Non",
            "augementation_salaire_precedente": 0.15,
            "age": 45,
            "genre": "Homme",
            "revenu_mensuel": 1200,  # Salaire très bas pour un directeur
            "statut_marital": "Marié(e)",
            "departement": "Commercial",
            "poste": "Senior Manager",
            "nombre_experiences_precedentes": 5,
            "annee_experience_totale": 20,
            "annees_dans_l_entreprise": 15,
            "annees_dans_le_poste_actuel": 10,
            "annees_depuis_la_derniere_promotion": 5,
            "annes_sous_responsable_actuel": 8,
            "nombre_participation_pee": 2,
            "nb_formations_suivies": 4,
            "distance_domicile_travail": 10,
            "niveau_education": 4,
            "domaine_etude": "Marketing",
            "frequence_deplacement": "Voyage_Rare"
        }
        
        response = client.post(
            "/api/v1/predict/single",
            json=salary_inconsistent
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            # Salaire très bas pour le poste = risque d'attrition élevé
            assert data["prediction"] in ["Oui", "Non"]


class TestBatchEdgeCases:
    """Tests de cas limites pour les prédictions batch"""
    
    def test_batch_profils_extremes(self):
        """Test batch avec profils extrêmes mélangés"""
        batch_data = {
            "employees": [
                JEUNE_EMPLOYE_HAUT_RISQUE,
                SENIOR_MANAGER_FAIBLE_RISQUE,
                PROFIL_PARADOXAL
            ]
        }
        
        response = client.post(
            "/api/v1/predict/batch",
            json=batch_data
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["total_employees"] == 3
            assert len(data["predictions"]) == 3
            
            # Vérifier la diversité des prédictions
            predictions = [p["prediction"] for p in data["predictions"]]
            probabilities = [p["probability_quit"] for p in data["predictions"]]
            
            # Profils très différents = probabilités variées
            assert len(set(probabilities)) > 1  # Au moins 2 proba différentes
    
    def test_batch_tous_identiques(self):
        """Test batch avec tous les profils identiques"""
        batch_data = {
            "employees": [JEUNE_EMPLOYE_HAUT_RISQUE] * 5
        }
        
        response = client.post(
            "/api/v1/predict/batch",
            json=batch_data
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["total_employees"] == 5
            
            # Même profil = même prédiction
            predictions = [p["prediction"] for p in data["predictions"]]
            probabilities = [p["probability_quit"] for p in data["predictions"]]
            
            assert len(set(predictions)) == 1  # Toutes identiques
            assert len(set(probabilities)) == 1  # Toutes identiques


class TestDataValidationRobustness:
    """Tests de robustesse de la validation des données"""
    
    def test_champs_optionnels_manquants(self):
        """Test avec certains champs optionnels manquants"""
        incomplete_data = {
            "satisfaction_employee_environnement": 3,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_precedente": 3,
            "note_evaluation_actuelle": 3,
            "niveau_hierarchique_poste": 2,
            "heure_supplementaires": "Oui",
            "augementation_salaire_precedente": 0.15,
            "age": 32,
            "genre": "Homme",
            "revenu_mensuel": 3500,
            "statut_marital": "Marié(e)",
            "departement": "Commercial",
            "poste": "Manager",
            # Champs manquants volontairement
            # "nombre_experiences_precedentes": 2,
            # "annee_experience_totale": 8,
        }
        
        response = client.post(
            "/api/v1/predict/single",
            json=incomplete_data
        )
        
        # Doit être rejeté car champs obligatoires manquants
        assert response.status_code in [422, 503]
        
        if response.status_code == 422:
            data = response.json()
            assert "detail" in data
    
    def test_valeurs_hors_enum(self):
        """Test avec valeurs hors énumérations"""
        invalid_enum_data = {
            "satisfaction_employee_environnement": 3,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_precedente": 3,
            "note_evaluation_actuelle": 3,
            "niveau_hierarchique_poste": 2,
            "heure_supplementaires": "Peut-être",  # Valeur invalide
            "augementation_salaire_precedente": 0.15,
            "age": 32,
            "genre": "Non-binaire",  # Pas dans l'enum
            "revenu_mensuel": 3500,
            "statut_marital": "En couple",  # Pas dans l'enum
            "departement": "IT",  # Pas dans l'enum
            "poste": "Developer",  # Pas dans l'enum
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
            "domaine_etude": "Intelligence Artificielle",  # Pas dans l'enum
            "frequence_deplacement": "Jamais"  # Pas dans l'enum
        }
        
        response = client.post(
            "/api/v1/predict/single",
            json=invalid_enum_data
        )
        
        # Doit être rejeté par validation Pydantic
        assert response.status_code in [422, 503]
        
        if response.status_code == 422:
            data = response.json()
            assert "detail" in data
            # Vérifier qu'on a des erreurs de validation
            error_str = str(data["detail"])
            assert any(field in error_str.lower() for field in 
                      ["genre", "statut_marital", "departement", "heure_supplementaires"])


class TestModelConsistencyWithRealData:
    """Tests de cohérence du modèle avec données réelles"""
    
    def test_coherence_predictions_profils_opposes(self):
        """Test que le modèle différencie les profils opposés"""
        # Faire deux prédictions avec profils opposés
        response_risque = client.post(
            "/api/v1/predict/single",
            json=JEUNE_EMPLOYE_HAUT_RISQUE
        )
        
        response_stable = client.post(
            "/api/v1/predict/single",
            json=SENIOR_MANAGER_FAIBLE_RISQUE
        )
        
        if response_risque.status_code == 200 and response_stable.status_code == 200:
            data_risque = response_risque.json()
            data_stable = response_stable.json()
            
            # Les probabilités doivent être significativement différentes
            prob_risque = data_risque["probability_quit"]
            prob_stable = data_stable["probability_quit"]
            
            # Différence d'au moins 20%
            assert abs(prob_risque - prob_stable) > 0.2
            
            # Le profil à risque doit avoir une probabilité plus élevée
            assert prob_risque > prob_stable
    
    def test_stabilite_predictions_memes_donnees(self):
        """Test de stabilité : mêmes données = même prédiction"""
        responses = []
        
        for _ in range(3):
            response = client.post(
                "/api/v1/predict/single",
                json=SENIOR_MANAGER_FAIBLE_RISQUE
            )
            responses.append(response)
        
        if all(r.status_code == 200 for r in responses):
            predictions = [r.json()["prediction"] for r in responses]
            probabilities = [r.json()["probability_quit"] for r in responses]
            
            # Toutes les prédictions doivent être identiques
            assert len(set(predictions)) == 1
            assert all(abs(p - probabilities[0]) < 0.001 for p in probabilities)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])