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

@router.get("/predictions/history")
async def get_prediction_history(
    limit: int = Query(50, le=200, description="Nombre de prédictions à récupérer"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Historique simple des prédictions récentes
    """
    try:
        with db_manager.get_session() as session:
            # Requête simple
            query = session.query(
                PredictionResult.prediction,
                PredictionResult.probability_quit,
                PredictionResult.confidence_level,
                PredictionResult.created_at
            ).order_by(desc(PredictionResult.created_at)).limit(limit)
            
            results = query.all()
            
            predictions = []
            for r in results:
                predictions.append({
                    "prediction": r.prediction,
                    "probability_quit": float(r.probability_quit),
                    "confidence_level": r.confidence_level,
                    "created_at": r.created_at.isoformat()
                })
            
            return {
                "predictions": predictions,
                "total_returned": len(predictions),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Erreur historique: {e}")
        raise HTTPException(status_code=500, detail=str(e))