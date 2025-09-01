"""
Router pour les endpoints de santé et de monitoring
Vérification de l'état de l'API et du modèle ML
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
import psutil
import time
from datetime import datetime

from app.models.schemas import ModelInfo
from app.models.ml_model import MLModel

router = APIRouter()

def get_ml_model() -> MLModel:
    """Dependency pour obtenir l'instance du modèle ML"""
    from app.main import app
    if not hasattr(app.state, 'ml_model') or app.state.ml_model is None:
        raise HTTPException(status_code=503, detail="Modèle ML non disponible")
    return app.state.ml_model

@router.get("/", response_model=Dict)
async def health_check():
    """
    Vérification de santé basique de l'API
    """
    return {
        "status": "healthy",
        "service": "Futurisys ML API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - psutil.boot_time() if psutil else None
    }

@router.get("/detailed", response_model=Dict)
async def detailed_health_check(ml_model: MLModel = Depends(get_ml_model)):
    """
    Vérification de santé détaillée incluant le modèle ML
    """
    # Santé du modèle
    model_health = ml_model.health_check()
    
    # Métriques système
    system_metrics = {}
    if psutil:
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    
    # Statut global
    overall_status = "healthy" if all(model_health.values()) else "degraded"
    
    return {
        "status": overall_status,
        "service": "Futurisys ML API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_health": model_health,
        "system_metrics": system_metrics,
        "checks": {
            "api_responsive": True,
            "model_loaded": model_health.get("model_loaded", False),
            "model_file_accessible": model_health.get("model_file_exists", False)
        }
    }

@router.get("/model", response_model=ModelInfo)
async def get_model_info(ml_model: MLModel = Depends(get_ml_model)):
    """
    Informations détaillées sur le modèle de ML
    """
    model_info = ml_model.get_model_info()
    
    return ModelInfo(
        model_name=model_info["model_name"],
        model_type=model_info["model_type"],
        version=model_info["version"],
        features_count=model_info["features_count"],
        training_date=model_info.get("training_date"),
        performance_metrics=model_info.get("performance_metrics", {}),
        threshold=model_info["threshold"]
    )

@router.get("/readiness")
async def readiness_check(ml_model: MLModel = Depends(get_ml_model)):
    """
    Vérification de préparation (readiness) pour Kubernetes/conteneurs
    """
    model_health = ml_model.health_check()
    
    if not all(model_health.values()):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service not ready",
                "issues": [k for k, v in model_health.items() if not v]
            }
        )
    
    return {"status": "ready"}

@router.get("/liveness")
async def liveness_check():
    """
    Vérification de vivacité (liveness) pour Kubernetes/conteneurs
    """
    # Test basique que l'API répond
    try:
        return {
            "status": "alive",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Liveness check failed: {str(e)}")

@router.post("/test-prediction")
async def test_prediction_endpoint(ml_model: MLModel = Depends(get_ml_model)):
    """
    Test rapide d'une prédiction avec des données factices
    Utile pour vérifier que le pipeline de prédiction fonctionne
    """
    from app.models.schemas import EmployeeData
    
    # Données de test
    test_employee = EmployeeData(
        satisfaction_employee_environnement=5,
        satisfaction_employee_nature_travail=6,
        satisfaction_employee_equipe=5,
        satisfaction_employee_equilibre_pro_perso=4,
        note_evaluation_precedente=3,
        note_evaluation_actuelle=3,
        niveau_hierarchique_poste=2,
        heure_supplementaires="Non",
        augementation_salaire_precedente="Non",
        age=30,
        genre="Homme",
        revenu_mensuel=3000,
                    statut_marital="Célibataire",
        departement="Recherche et Développement",
        poste="Développeur",
        nombre_experiences_precedentes=1,
        annee_experience_totale=5,
        annees_dans_l_entreprise=2,
        annees_dans_le_poste_actuel=1,
        annees_depuis_la_derniere_promotion=2,
        annes_sous_responsable_actuel=1,
        nombre_participation_pee=0,
        nb_formations_suivies=2,
        distance_domicile_travail=10,
        niveau_education=3,
        domaine_etude="Informatique",
        frequence_deplacement="Voyage_Rare"
    )
    
    try:
        start_time = time.time()
        prediction = ml_model.predict_single(test_employee, employee_id="test_emp")
        processing_time = time.time() - start_time
        
        return {
            "test_status": "success",
            "prediction_result": prediction.dict(),
            "processing_time_seconds": round(processing_time, 4),
            "message": "Prédiction de test réussie"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "test_status": "failed",
                "error": str(e),
                "message": "Échec du test de prédiction"
            }
        )