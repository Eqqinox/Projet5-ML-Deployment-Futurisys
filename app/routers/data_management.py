"""
Router pour la consultation et gestion des données PostgreSQL
Endpoints pour consulter les employés, sessions et historique des prédictions
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_

from app.database.connection import DatabaseManager
from app.database.models import Employee, PredictionSession, PredictionRequest, PredictionResult, ModelMetadata
from app.models.schemas import EmployeeData

router = APIRouter()
logger = logging.getLogger(__name__)

def get_database_manager() -> DatabaseManager:
    """Dependency pour obtenir l'instance du DatabaseManager"""
    from app.main import app
    if not hasattr(app.state, 'db_manager') or app.state.db_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    return app.state.db_manager

@router.get("/employees/count")
async def get_employees_count(db_manager: DatabaseManager = Depends(get_database_manager)):
    """
    Nombre total d'employés dans la base de données
    """
    try:
        with db_manager.get_session() as session:
            total_employees = session.query(Employee).count()
            
            # Statistiques par département
            dept_stats = session.query(
                Employee.departement,
                func.count(Employee.employee_id).label('count')
            ).group_by(Employee.departement).all()
            
            # Statistiques par statut d'attrition
            attrition_stats = session.query(
                Employee.a_quitte_l_entreprise,
                func.count(Employee.employee_id).label('count')
            ).group_by(Employee.a_quitte_l_entreprise).all()
            
            return {
                "total_employees": total_employees,
                "by_department": {dept: count for dept, count in dept_stats},
                "by_attrition_status": {status: count for status, count in attrition_stats},
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Erreur lors du comptage des employés: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/employees/{employee_id}")
async def get_employee_details(
    employee_id: int,
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Détails d'un employé spécifique avec son historique de prédictions
    """
    try:
        with db_manager.get_session() as session:
            # Récupération de l'employé
            employee = session.query(Employee).filter_by(employee_id=employee_id).first()
            
            if not employee:
                raise HTTPException(status_code=404, detail="Employé non trouvé")
            
            # Historique des prédictions pour cet employé
            predictions_history = session.query(
                PredictionRequest.created_at,
                PredictionResult.prediction,
                PredictionResult.probability_quit,
                PredictionResult.confidence_level,
                PredictionResult.model_version
            ).join(
                PredictionResult, PredictionRequest.request_id == PredictionResult.request_id
            ).filter(
                PredictionRequest.employee_id == employee_id
            ).order_by(desc(PredictionRequest.created_at)).limit(10).all()
            
            predictions_list = []
            for pred in predictions_history:
                predictions_list.append({
                    "date": pred.created_at.isoformat(),
                    "prediction": pred.prediction,
                    "probability_quit": float(pred.probability_quit),
                    "confidence_level": pred.confidence_level,
                    "model_version": pred.model_version
                })
            
            return {
                "employee": employee.to_dict(),
                "predictions_history": predictions_list,
                "total_predictions": len(predictions_list)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'employé {employee_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/employees/search")
async def search_employees(
    department: Optional[str] = Query(None, description="Filtrer par département"),
    age_min: Optional[int] = Query(None, description="Âge minimum"),
    age_max: Optional[int] = Query(None, description="Âge maximum"),
    attrition_status: Optional[str] = Query(None, description="Statut d'attrition (Oui/Non)"),
    limit: int = Query(50, le=500, description="Nombre maximum de résultats"),
    offset: int = Query(0, description="Décalage pour la pagination"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Recherche d'employés avec filtres
    """
    try:
        with db_manager.get_session() as session:
            query = session.query(Employee)
            
            # Application des filtres
            if department:
                query = query.filter(Employee.departement == department)
            if age_min:
                query = query.filter(Employee.age >= age_min)
            if age_max:
                query = query.filter(Employee.age <= age_max)
            if attrition_status:
                query = query.filter(Employee.a_quitte_l_entreprise == attrition_status)
            
            # Pagination
            total_count = query.count()
            employees = query.offset(offset).limit(limit).all()
            
            return {
                "employees": [emp.to_dict() for emp in employees],
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "returned": len(employees)
                },
                "filters_applied": {
                    "department": department,
                    "age_min": age_min,
                    "age_max": age_max,
                    "attrition_status": attrition_status
                }
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la recherche d'employés: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def get_prediction_sessions(
    session_type: Optional[str] = Query(None, description="Type de session (single/batch)"),
    status: Optional[str] = Query(None, description="Statut de la session"),
    days_back: int = Query(7, description="Nombre de jours à consulter"),
    limit: int = Query(100, le=1000),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Liste des sessions de prédiction avec filtres
    """
    try:
        with db_manager.get_session() as session:
            # Date limite
            date_limit = datetime.now() - timedelta(days=days_back)
            
            query = session.query(PredictionSession).filter(
                PredictionSession.started_at >= date_limit
            )
            
            # Filtres
            if session_type:
                query = query.filter(PredictionSession.session_type == session_type)
            if status:
                query = query.filter(PredictionSession.status == status)
            
            # Tri et limitation
            sessions = query.order_by(desc(PredictionSession.started_at)).limit(limit).all()
            
            # Statistiques
            stats = {
                "total_sessions": query.count(),
                "by_type": {},
                "by_status": {}
            }
            
            # Comptage par type
            type_counts = session.query(
                PredictionSession.session_type,
                func.count(PredictionSession.session_id)
            ).filter(
                PredictionSession.started_at >= date_limit
            ).group_by(PredictionSession.session_type).all()
            stats["by_type"] = {t: c for t, c in type_counts}
            
            # Comptage par statut
            status_counts = session.query(
                PredictionSession.status,
                func.count(PredictionSession.session_id)
            ).filter(
                PredictionSession.started_at >= date_limit
            ).group_by(PredictionSession.status).all()
            stats["by_status"] = {s: c for s, c in status_counts}
            
            return {
                "sessions": [s.to_dict() for s in sessions],
                "statistics": stats,
                "period": f"Last {days_back} days",
                "returned": len(sessions)
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/details")
async def get_session_details(
    session_id: str,
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Détails complets d'une session de prédiction
    """
    try:
        with db_manager.get_session() as session:
            # Session principale
            pred_session = session.query(PredictionSession).filter_by(
                session_id=session_id
            ).first()
            
            if not pred_session:
                raise HTTPException(status_code=404, detail="Session non trouvée")
            
            # Requêtes de la session
            requests = session.query(PredictionRequest).filter_by(
                session_id=session_id
            ).all()
            
            # Résultats avec jointure
            results_query = session.query(
                PredictionRequest.request_id,
                PredictionRequest.input_data,
                PredictionRequest.created_at,
                PredictionResult.prediction,
                PredictionResult.probability_quit,
                PredictionResult.confidence_level,
                PredictionResult.risk_factors
            ).join(
                PredictionResult, PredictionRequest.request_id == PredictionResult.request_id
            ).filter(PredictionRequest.session_id == session_id)
            
            results = []
            for result in results_query.all():
                results.append({
                    "request_id": result.request_id,
                    "input_data": result.input_data,
                    "created_at": result.created_at.isoformat(),
                    "prediction": result.prediction,
                    "probability_quit": float(result.probability_quit),
                    "confidence_level": result.confidence_level,
                    "risk_factors": result.risk_factors
                })
            
            return {
                "session": pred_session.to_dict(),
                "requests": [req.to_dict() for req in requests],
                "results": results,
                "summary": {
                    "total_requests": len(requests),
                    "total_results": len(results),
                    "completion_rate": len(results) / len(requests) if requests else 0
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des détails de session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def delete_prediction_session(
    session_id: str,
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Suppression d'une session de prédiction (cascade sur requests/results)
    """
    try:
        with db_manager.get_session() as session:
            pred_session = session.query(PredictionSession).filter_by(
                session_id=session_id
            ).first()
            
            if not pred_session:
                raise HTTPException(status_code=404, detail="Session non trouvée")
            
            # Compter les éléments avant suppression
            requests_count = session.query(PredictionRequest).filter_by(
                session_id=session_id
            ).count()
            
            # Suppression (cascade automatique sur requests/results)
            session.delete(pred_session)
            
            return {
                "message": f"Session {session_id} supprimée avec succès",
                "deleted_requests": requests_count,
                "timestamp": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/recent")
async def get_recent_predictions(
    hours_back: int = Query(24, description="Nombre d'heures à consulter"),
    prediction_filter: Optional[str] = Query(None, description="Filtrer par prédiction (Oui/Non)"),
    confidence_min: Optional[str] = Query(None, description="Niveau de confiance minimum"),
    limit: int = Query(100, le=500),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Prédictions récentes avec statistiques
    """
    try:
        with db_manager.get_session() as session:
            # Date limite
            date_limit = datetime.now() - timedelta(hours=hours_back)
            
            # Requête principale avec jointures
            query = session.query(
                PredictionSession.session_id,
                PredictionSession.session_type,
                PredictionSession.started_at,
                PredictionRequest.input_data,
                PredictionResult.prediction,
                PredictionResult.probability_quit,
                PredictionResult.confidence_level,
                PredictionResult.risk_factors,
                PredictionResult.model_version,
                PredictionResult.created_at
            ).join(
                PredictionRequest, PredictionSession.session_id == PredictionRequest.session_id
            ).join(
                PredictionResult, PredictionRequest.request_id == PredictionResult.request_id
            ).filter(
                PredictionResult.created_at >= date_limit
            )
            
            # Filtres
            if prediction_filter:
                query = query.filter(PredictionResult.prediction == prediction_filter)
            if confidence_min:
                query = query.filter(PredictionResult.confidence_level == confidence_min)
            
            # Récupération des résultats
            results = query.order_by(desc(PredictionResult.created_at)).limit(limit).all()
            
            predictions_list = []
            for result in results:
                predictions_list.append({
                    "session_id": str(result.session_id),
                    "session_type": result.session_type,
                    "session_started": result.started_at.isoformat(),
                    "prediction": result.prediction,
                    "probability_quit": float(result.probability_quit),
                    "confidence_level": result.confidence_level,
                    "risk_factors": result.risk_factors,
                    "model_version": result.model_version,
                    "predicted_at": result.created_at.isoformat()
                })
            
            # Statistiques de la période
            total_predictions = query.count()
            quit_predictions = query.filter(PredictionResult.prediction == "Oui").count()
            stay_predictions = total_predictions - quit_predictions
            
            # Moyenne des probabilités
            avg_quit_prob = session.query(
                func.avg(PredictionResult.probability_quit)
            ).join(
                PredictionRequest, PredictionRequest.request_id == PredictionResult.request_id
            ).filter(
                PredictionResult.created_at >= date_limit
            ).scalar()
            
            return {
                "predictions": predictions_list,
                "statistics": {
                    "total_predictions": total_predictions,
                    "quit_predictions": quit_predictions,
                    "stay_predictions": stay_predictions,
                    "quit_rate": quit_predictions / total_predictions if total_predictions > 0 else 0,
                    "average_quit_probability": float(avg_quit_prob) if avg_quit_prob else 0
                },
                "period": f"Last {hours_back} hours",
                "returned": len(predictions_list),
                "filters_applied": {
                    "prediction": prediction_filter,
                    "confidence_min": confidence_min
                }
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des prédictions récentes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/metadata")
async def get_model_metadata(
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Métadonnées du modèle actuel et historique des versions
    """
    try:
        with db_manager.get_session() as session:
            # Modèle actuel
            current_model = session.query(ModelMetadata).filter_by(is_active=True).first()
            
            # Historique des modèles
            model_history = session.query(ModelMetadata).order_by(
                desc(ModelMetadata.created_at)
            ).all()
            
            return {
                "current_model": current_model.to_dict() if current_model else None,
                "model_history": [model.to_dict() for model in model_history],
                "total_versions": len(model_history),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métadonnées: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database/stats")
async def get_database_statistics(
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Statistiques générales de la base de données
    """
    try:
        with db_manager.get_session() as session:
            stats = {
                "tables": {
                    "employees": session.query(Employee).count(),
                    "prediction_sessions": session.query(PredictionSession).count(),
                    "prediction_requests": session.query(PredictionRequest).count(),
                    "prediction_results": session.query(PredictionResult).count(),
                    "model_metadata": session.query(ModelMetadata).count()
                },
                "recent_activity": {},
                "performance": {}
            }
            
            # Activité récente (dernières 24h)
            last_24h = datetime.now() - timedelta(hours=24)
            
            stats["recent_activity"] = {
                "new_sessions_24h": session.query(PredictionSession).filter(
                    PredictionSession.started_at >= last_24h
                ).count(),
                "new_predictions_24h": session.query(PredictionResult).filter(
                    PredictionResult.created_at >= last_24h
                ).count()
            }
            
            # Calcul des performances moyennes
            avg_processing_time = session.query(
                func.avg(PredictionResult.processing_time_ms)
            ).scalar()
            
            stats["performance"] = {
                "avg_processing_time_ms": float(avg_processing_time) if avg_processing_time else None,
                "total_processed_predictions": stats["tables"]["prediction_results"]
            }
            
            return {
                "database_statistics": stats,
                "generated_at": datetime.now().isoformat(),
                "health_status": "healthy"
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques: {e}")
        raise HTTPException(status_code=500, detail=str(e))