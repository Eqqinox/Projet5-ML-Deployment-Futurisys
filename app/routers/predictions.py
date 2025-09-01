"""
Router pour les endpoints de prédiction d'attrition des employés
Endpoints pour prédictions individuelles et batch
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
import time
import logging
from datetime import datetime

from app.models.schemas import (
    EmployeeData, 
    BatchEmployeeData,
    PredictionResult, 
    BatchPredictionResult,
    ErrorResponse
)
from app.models.ml_model import MLModel
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def get_ml_model() -> MLModel:
    """Dependency pour obtenir l'instance du modèle ML"""
    from app.main import app
    if not hasattr(app.state, 'ml_model') or app.state.ml_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle ML non disponible. Veuillez vérifier le statut avec /health"
        )
    return app.state.ml_model

@router.post("/predict/single", response_model=PredictionResult)
async def predict_single_employee(
    employee: EmployeeData,
    ml_model: MLModel = Depends(get_ml_model)
):
    """
    Prédiction d'attrition pour un seul employé
    
    - **employee**: Données complètes de l'employé
    - **return**: Résultat de prédiction avec probabilités et facteurs de risque
    """
    try:
        logger.info(f"Nouvelle demande de prédiction individuelle")
        
        start_time = time.time()
        
        # Validation implicite via Pydantic déjà effectuée
        prediction = ml_model.predict_single(employee)
        
        processing_time = time.time() - start_time
        logger.info(f"Prédiction réalisée en {processing_time:.4f}s - Résultat: {prediction.prediction}")
        
        return prediction
        
    except ValueError as e:
        logger.error(f"Erreur de validation des données: {e}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": f"Données d'entrée invalides: {str(e)}",
                "details": {"validation_issue": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction Error",
                "message": "Erreur interne lors de la prédiction",
                "details": {"error_type": type(e).__name__, "error_message": str(e)}
            }
        )

@router.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch_employees(
    batch_data: BatchEmployeeData,
    ml_model: MLModel = Depends(get_ml_model)
):
    """
    Prédictions d'attrition pour plusieurs employés simultanément
    
    - **batch_data**: Liste d'employés (maximum 100)
    - **return**: Résultats de prédictions avec statistiques aggregées
    """
    try:
        logger.info(f"Nouvelle demande de prédiction batch pour {len(batch_data.employees)} employés")
        
        # Vérification de la taille du batch
        if len(batch_data.employees) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Batch Size Error", 
                    "message": f"Taille du batch trop importante. Maximum autorisé: {settings.MAX_BATCH_SIZE}",
                    "current_size": len(batch_data.employees),
                    "max_size": settings.MAX_BATCH_SIZE
                }
            )
        
        start_time = time.time()
        
        # Prédictions
        predictions = ml_model.predict_batch(batch_data.employees)
        
        processing_time = time.time() - start_time
        
        # Calcul des statistiques
        quit_predictions = len([p for p in predictions if p.prediction == "Oui"])
        stay_predictions = len([p for p in predictions if p.prediction == "Non"])
        average_quit_probability = sum([p.probability_quit for p in predictions]) / len(predictions)
        
        logger.info(f"Batch de {len(predictions)} prédictions réalisées en {processing_time:.4f}s")
        logger.info(f"Résultats: {quit_predictions} départs prédits, {stay_predictions} rétentions")
        
        return BatchPredictionResult(
            predictions=predictions,
            total_employees=len(predictions),
            quit_predictions=quit_predictions,
            stay_predictions=stay_predictions,
            average_quit_probability=round(average_quit_probability, 4),
            processing_time_seconds=round(processing_time, 4)
        )
        
    except HTTPException:
        # Re-raise HTTPException sans modification
        raise
    except ValueError as e:
        logger.error(f"Erreur de validation batch: {e}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Batch Validation Error",
                "message": f"Erreur dans les données du batch: {str(e)}",
                "details": {"validation_issue": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction batch: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Batch Prediction Error",
                "message": "Erreur interne lors du traitement du batch",
                "details": {"error_type": type(e).__name__, "error_message": str(e)}
            }
        )

@router.get("/predict/model-info")
async def get_prediction_model_info(ml_model: MLModel = Depends(get_ml_model)):
    """
    Informations détaillées sur le modèle de prédiction
    """
    try:
        model_info = ml_model.get_model_info()
        return {
            "model_information": model_info,
            "endpoint_capabilities": {
                "single_prediction": True,
                "batch_prediction": True,
                "max_batch_size": settings.MAX_BATCH_SIZE,
                "supported_features": ml_model.features_order,
                "prediction_timeout": settings.PREDICTION_TIMEOUT
            },
            "output_format": {
                "prediction": "Oui/Non (attrition)",
                "probabilities": "probability_quit et probability_stay (0-1)",
                "confidence_levels": ["Faible", "Moyen", "Élevé"],
                "risk_factors": "Liste des facteurs de risque identifiés"
            }
        }
    except Exception as e:
        logger.error(f"Erreur récupération info modèle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/validate-input", response_model=dict)
async def validate_employee_input(employee: EmployeeData):
    """
    Validation des données d'entrée sans effectuer de prédiction
    Utile pour tester la conformité des données avant envoi
    """
    try:
        # La validation Pydantic s'exécute automatiquement
        return {
            "validation_status": "success",
            "message": "Données d'employé valides",
            "received_data": employee.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={
                "validation_status": "failed",
                "message": f"Données invalides: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@router.get("/predict/supported-values")
async def get_supported_categorical_values():
    """
    Liste des valeurs supportées pour les variables catégorielles
    """
    return {
        "categorical_variables": {
            "heure_supplementaires": ["Oui", "Non"],
            "genre": ["Homme", "Femme"],
            "statut_marital": ["Célibataire", "Marié(e)", "Divorcé(e)"],
            "departement": ["Commercial", "Consulting", "Ressources Humaines"],
            "poste": ["Cadre Commercial", "Assistant de Direction", "Consultant", 
                     "Tech Lead", "Manager", "Senior Manager", "Représentant Commercial",
                     "Directeur Technique", "Ressources Humaines"],
            "domaine_etude": ["Infra & Cloud", "Autre", "Transformation Digitale", 
                             "Marketing", "Entrepreneuriat", "Ressources Humaines"],
            "frequence_deplacement": ["Voyage_Fréquent", "Voyage_Rare", "Pas_de_Voyage"]
        },
        "numerical_ranges": {
            "satisfaction_scores": {"min": 1, "max": 4, "description": "Scores de satisfaction"},
            "evaluation_scores": {"min": 1, "max": 4, "description": "Notes d'évaluation"},
            "niveau_hierarchique": {"min": 1, "max": 5},
            "age": {"min": 18, "max": 65, "description": "43 valeurs différentes possibles"},
            "revenu_mensuel": {"min": 1000, "max": 50000, "description": "1349 valeurs différentes possibles"},
            "niveau_education": {"min": 1, "max": 5},
            "augementation_salaire_precedente": {"min": 0.11, "max": 0.25, "description": "Pourcentage d'augmentation"},
            "nombre_experiences_precedentes": {"min": 0, "max": 9, "description": "10 valeurs différentes"},
            "annee_experience_totale": {"min": 0, "max": 40, "description": "40 valeurs différentes"},
            "annees_dans_l_entreprise": {"min": 0, "max": 36, "description": "37 valeurs différentes"},
            "annees_dans_le_poste_actuel": {"min": 0, "max": 18, "description": "19 valeurs différentes"},
            "nombre_participation_pee": {"min": 0, "max": 3},
            "nb_formations_suivies": {"min": 0, "max": 6},
            "distance_domicile_travail": {"min": 1, "max": 29, "description": "29 valeurs différentes"},
            "annees_depuis_la_derniere_promotion": {"min": 0, "max": 15, "description": "16 valeurs différentes"},
            "annes_sous_responsable_actuel": {"min": 0, "max": 17, "description": "18 valeurs différentes"}
        },
        "note": "Les valeurs pour 'departement', 'poste' et 'domaine_etude' dépendent de l'encodage du modèle"
    }