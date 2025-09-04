"""
Router pour les endpoints de prédiction d'attrition des employés avec traçabilité PostgreSQL
Endpoints pour prédictions individuelles et batch avec sauvegarde automatique des inputs/outputs
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
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
from app.middleware.prediction_logger import PredictionSessionManager

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

def get_database_manager():
    """Dependency pour obtenir l'instance du DatabaseManager"""
    from app.main import app
    if not hasattr(app.state, 'db_manager') or app.state.db_manager is None:
        logger.warning("DatabaseManager non disponible - traçabilité désactivée")
        return None
    return app.state.db_manager

@router.post("/predict/single", response_model=PredictionResult)
async def predict_single_employee(
    request: Request,
    employee: EmployeeData,
    ml_model: MLModel = Depends(get_ml_model),
    db_manager = Depends(get_database_manager)
):
    """
    Prédiction d'attrition pour un seul employé avec traçabilité complète
    
    - **employee**: Données complètes de l'employé
    - **return**: Résultat de prédiction avec probabilités et facteurs de risque
    
    **Traçabilité**: Tous les inputs et outputs sont automatiquement sauvegardés en base PostgreSQL
    """
    request_id = None
    
    try:
        logger.info("Nouvelle demande de prédiction individuelle")
        
        start_time = time.time()
        
        # 1. Sauvegarde de l'input en base de données (si DB disponible)
        if db_manager:
            try:
                # Convertir les données Pydantic en dict pour la sauvegarde
                input_data = employee.dict()
                
                # Tentative de matching avec un employé existant en base
                existing_employee = db_manager.get_employee_by_data(input_data)
                employee_id = existing_employee.employee_id if existing_employee else None
                
                # Sauvegarde de la requête
                request_id = await PredictionSessionManager.save_prediction_input(
                    request=request,
                    input_data=input_data,
                    employee_id=employee_id
                )
                
                if request_id:
                    logger.debug(f"Input sauvegardé en base: request_id={request_id}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde de l'input: {e}")
                # Continuer sans traçabilité plutôt que de faire crash
        
        # 2. Exécution de la prédiction (logique métier inchangée)
        prediction = ml_model.predict_single(employee)
        
        processing_time = time.time() - start_time
        
        # 3. Sauvegarde de l'output en base de données (si DB disponible et request_id existe)
        if db_manager and request_id:
            try:
                result_id = await PredictionSessionManager.save_prediction_output(
                    request=request,
                    request_id=request_id,
                    prediction_result=prediction
                )
                
                if result_id:
                    logger.debug(f"Output sauvegardé en base: result_id={result_id}")
                
                # Marquer la session comme terminée
                await PredictionSessionManager.complete_session(request, total_predictions=1)
                
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde de l'output: {e}")
        
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
    request: Request,
    batch_data: BatchEmployeeData,
    ml_model: MLModel = Depends(get_ml_model),
    db_manager = Depends(get_database_manager)
):
    """
    Prédictions d'attrition pour plusieurs employés simultanément avec traçabilité
    
    - **batch_data**: Liste d'employés (maximum 100)
    - **return**: Résultats de prédictions avec statistiques aggregées
    
    **Traçabilité**: Chaque employé du batch est traçé individuellement en base PostgreSQL
    """
    request_ids = []
    predictions = []
    
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
        
        # Traitement de chaque employé du batch
        for i, employee in enumerate(batch_data.employees):
            employee_request_id = None
            
            try:
                # 1. Sauvegarde de l'input pour cet employé
                if db_manager:
                    try:
                        input_data = employee.dict()
                        existing_employee = db_manager.get_employee_by_data(input_data)
                        employee_id = existing_employee.employee_id if existing_employee else None
                        
                        employee_request_id = await PredictionSessionManager.save_prediction_input(
                            request=request,
                            input_data=input_data,
                            employee_id=employee_id
                        )
                        
                        if employee_request_id:
                            request_ids.append(employee_request_id)
                        
                    except Exception as e:
                        logger.error(f"Erreur sauvegarde input employé {i+1}: {e}")
                
                # 2. Prédiction pour cet employé
                prediction = ml_model.predict_single(employee)
                predictions.append(prediction)
                
                # 3. Sauvegarde de l'output pour cet employé
                if db_manager and employee_request_id:
                    try:
                        result_id = await PredictionSessionManager.save_prediction_output(
                            request=request,
                            request_id=employee_request_id,
                            prediction_result=prediction
                        )
                        
                        logger.debug(f"Employé {i+1}: request_id={employee_request_id}, result_id={result_id}")
                        
                    except Exception as e:
                        logger.error(f"Erreur sauvegarde output employé {i+1}: {e}")
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de l'employé {i+1}: {e}")
                # Continuer avec les autres employés plutôt que de tout faire crash
                continue
        
        processing_time = time.time() - start_time
        
        # Marquer la session batch comme terminée
        if db_manager:
            try:
                await PredictionSessionManager.complete_session(request, total_predictions=len(predictions))
            except Exception as e:
                logger.error(f"Erreur completion session batch: {e}")
        
        # Calcul des statistiques (logique métier inchangée)
        quit_predictions = len([p for p in predictions if p.prediction == "Oui"])
        stay_predictions = len([p for p in predictions if p.prediction == "Non"])
        average_quit_probability = sum([p.probability_quit for p in predictions]) / len(predictions) if predictions else 0
        
        logger.info(f"Batch de {len(predictions)} prédictions réalisées en {processing_time:.4f}s")
        logger.info(f"Résultats: {quit_predictions} départs prédits, {stay_predictions} rétentions")
        logger.info(f"Traçabilité: {len(request_ids)} inputs sauvegardés en base")
        
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
                "traceability_enabled": True,  # Nouveau: traçabilité
                "database_integration": True,  # Nouveau: intégration BDD
                "supported_features": getattr(ml_model, 'features_order', []),
                "prediction_timeout": getattr(settings, 'PREDICTION_TIMEOUT', 30)
            },
            "output_format": {
                "prediction": "Oui/Non (attrition)",
                "probabilities": "probability_quit et probability_stay (0-1)",
                "confidence_levels": ["Faible", "Moyen", "Élevé"],
                "risk_factors": "Liste des facteurs de risque identifiés"
            },
            "traceability_info": {  # Nouveau: info traçabilité
                "input_storage": "Toutes les données d'entrée sont sauvegardées",
                "output_storage": "Tous les résultats de prédiction sont sauvegardés",
                "session_tracking": "Suivi complet des sessions single/batch",
                "employee_linking": "Association automatique avec les employés en base"
            }
        }
    except Exception as e:
        logger.error(f"Erreur récupération info modèle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/validate-input", response_model=dict)
async def validate_employee_input(
    request: Request,
    employee: EmployeeData,
    db_manager = Depends(get_database_manager)
):
    """
    Validation des données d'entrée sans effectuer de prédiction
    Utile pour tester la conformité des données avant envoi
    
    **Traçabilité**: La validation est également tracée en base
    """
    try:
        # Validation Pydantic automatique
        employee_data = employee.dict()
        
        # Sauvegarde optionnelle de la validation (sans prédiction)
        if db_manager:
            try:
                # Marquer comme validation seulement
                validation_data = {
                    **employee_data,
                    "_validation_only": True,
                    "_timestamp": datetime.now().isoformat()
                }
                
                request_id = await PredictionSessionManager.save_prediction_input(
                    request=request,
                    input_data=validation_data,
                    employee_id=None
                )
                
                logger.debug(f"Validation tracée: request_id={request_id}")
                
            except Exception as e:
                logger.warning(f"Erreur traçabilité validation: {e}")
        
        return {
            "validation_status": "success",
            "message": "Données d'employé valides",
            "received_data": employee_data,
            "traceability_enabled": db_manager is not None,
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
            "genre": ["F", "M"],  # Correction: F/M au lieu de Homme/Femme
            "statut_marital": ["Célibataire", "Marié(e)", "Divorcé(e)"],
            "departement": ["Commercial", "Consulting", "Ressources Humaines"],
            "poste": ["Cadre Commercial", "Assistant de Direction", "Consultant", 
                     "Tech Lead", "Manager", "Senior Manager", "Représentant Commercial",
                     "Directeur Technique", "Ressources Humaines"],
            "domaine_etude": ["Infra & Cloud", "Autre", "Transformation Digitale", 
                             "Marketing", "Entrepreunariat", "Ressources Humaines"],
            "frequence_deplacement": ["Aucun", "Occasionnel", "Frequent"]  # Correction: valeurs du dataset
        },
        "numerical_ranges": {
            "satisfaction_scores": {"min": 1, "max": 4, "description": "Scores de satisfaction"},
            "evaluation_scores": {"min": 1, "max": 4, "description": "Notes d'évaluation"},
            "niveau_hierarchique": {"min": 1, "max": 5},
            "age": {"min": 18, "max": 60, "description": "Plage réelle du dataset"},
            "revenu_mensuel": {"min": 1009, "max": 19999, "description": "Plage réelle du dataset"},
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
        "database_info": {
            "total_employees_in_dataset": 1470,
            "traceability_active": True,
            "supported_encoding": "OneHot + Ordinal comme dans le Projet 4"
        }
    }

# Nouvel endpoint: Historique des prédictions
@router.get("/predict/history")
async def get_prediction_history(
    limit: int = 100,
    db_manager = Depends(get_database_manager)
):
    """
    Récupère l'historique des prédictions depuis la base de données
    
    **Nécessite**: Base de données PostgreSQL connectée
    """
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible - Historique inaccessible"
        )
    
    try:
        history = db_manager.get_prediction_history(limit=min(limit, 1000))  # Max 1000
        
        return {
            "history": history,
            "total_returned": len(history),
            "limit_applied": limit,
            "timestamp": datetime.now().isoformat(),
            "note": "Historique complet des prédictions avec traçabilité"
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération historique: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "History Retrieval Error",
                "message": "Erreur lors de la récupération de l'historique",
                "details": str(e)
            }
        )