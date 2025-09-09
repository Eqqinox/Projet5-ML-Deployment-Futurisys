"""
Router pour les endpoints de prédiction d'attrition des employés
Endpoints pour prédictions individuelles avec validation
"""

from fastapi import APIRouter, Depends, HTTPException
import logging
from datetime import datetime

from app.models.schemas import (
    EmployeeData, 
    PredictionResult,
    BatchEmployeeData,
    BatchPredictionResult
)
from app.models.ml_model import MLModel

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
        logger.info("Nouvelle demande de prédiction individuelle")
        
        # Exécution de la prédiction
        prediction = ml_model.predict_single(employee)
        
        logger.info(f"Prédiction réalisée - Résultat: {prediction.prediction}")
        
        return prediction
        
    except ValueError as e:
        logger.error(f"Erreur de validation des données: {e}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": f"Données d'entrée invalides: {str(e)}"
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction Error",
                "message": "Erreur interne lors de la prédiction"
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
    - **return**: Résultats de prédictions avec statistiques
    """
    try:
        logger.info(f"Prédiction batch pour {len(batch_data.employees)} employés")
        
        # Limite de 100 employés
        if len(batch_data.employees) > 100:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Batch trop volumineux",
                    "message": "Maximum 100 employés autorisés",
                    "current_size": len(batch_data.employees)
                }
            )
        
        # Prédictions pour chaque employé
        predictions = []
        for employee in batch_data.employees:
            prediction = ml_model.predict_single(employee)
            predictions.append(prediction)
        
        # Statistiques
        quit_count = len([p for p in predictions if p.prediction == "Oui"])
        stay_count = len([p for p in predictions if p.prediction == "Non"])
        avg_quit_prob = sum([p.probability_quit for p in predictions]) / len(predictions)
        
        return BatchPredictionResult(
            predictions=predictions,
            total_employees=len(predictions),
            quit_predictions=quit_count,
            stay_predictions=stay_count,
            average_quit_probability=round(avg_quit_prob, 4)
        )
        
    except Exception as e:
        logger.error(f"Erreur prédiction batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/validate-input", response_model=dict)
async def validate_employee_input(employee: EmployeeData):
    """
    Validation des données d'entrée sans effectuer de prédiction
    Utile pour tester la conformité des données avant envoi
    """
    try:
        # Validation Pydantic automatique
        employee_data = employee.dict()
        
        return {
            "validation_status": "success",
            "message": "Données d'employé valides",
            "received_data": employee_data,
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
            "genre": ["Femme", "Homme"],
            "statut_marital": ["Célibataire", "Marié(e)", "Divorcé(e)"],
            "departement": ["Commercial", "Consulting", "Ressources Humaines"],
            "poste": ["Cadre Commercial", "Assistant de Direction", "Consultant", 
                     "Tech Lead", "Manager", "Senior Manager", "Représentant Commercial",
                     "Directeur Technique", "Ressources Humaines"],
            "domaine_etude": ["Infra & Cloud", "Autre", "Transformation Digitale", "Marketing", "Entrepreneuriat", "Ressources Humaines"],
            "frequence_deplacement": ["Aucun", "Voyage_Rare", "Voyage_Fréquent"]
        },
        "numerical_ranges": {
            "satisfaction_scores": {"min": 1, "max": 4, "description": "Scores de satisfaction"},
            "evaluation_scores": {"min": 1, "max": 4, "description": "Notes d'évaluation"},
            "niveau_hierarchique": {"min": 1, "max": 5},
            "age": {"min": 18, "max": 60},
            "revenu_mensuel": {"min": 1009, "max": 19999},
            "niveau_education": {"min": 1, "max": 5},
            "augementation_salaire_precedente": {"min": 0.11, "max": 0.25}
        },
        "model_info": {
            "total_features": 27,
            "algorithm": "XGBoost Classifier",
            "dataset_size": 1470
        }
    }