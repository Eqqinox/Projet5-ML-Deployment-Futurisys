"""
Router pour les analytics de base sur les prédictions
Endpoint pour les statistiques des prédictions
"""

from fastapi import APIRouter, Depends, HTTPException, Query
import logging
from datetime import datetime, timedelta
from sqlalchemy import func

from app.database.connection import DatabaseManager
from app.database.models import PredictionResult

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

@router.get("/predictions/stats")
async def get_predictions_stats(
    days_back: int = Query(30, le=90, description="Période d'analyse en jours"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Statistiques simples sur les prédictions récentes
    """
    try:
        with db_manager.get_session() as session:
            date_limit = datetime.now() - timedelta(days=days_back)
            
            # Statistiques de base
            total_predictions = session.query(PredictionResult).filter(
                PredictionResult.created_at >= date_limit
            ).count()
            
            # Distribution des prédictions
            quit_predictions = session.query(PredictionResult).filter(
                PredictionResult.created_at >= date_limit,
                PredictionResult.prediction == "Oui"
            ).count()
            
            stay_predictions = total_predictions - quit_predictions
            
            # Probabilité moyenne
            avg_quit_probability = session.query(
                func.avg(PredictionResult.probability_quit)
            ).filter(
                PredictionResult.created_at >= date_limit
            ).scalar()
            
            # Distribution par niveau de confiance
            confidence_dist = session.query(
                PredictionResult.confidence_level,
                func.count(PredictionResult.result_id).label('count')
            ).filter(
                PredictionResult.created_at >= date_limit
            ).group_by(PredictionResult.confidence_level).all()
            
            confidence_stats = {level: count for level, count in confidence_dist}
            
            return {
                "period": f"Last {days_back} days",
                "total_predictions": total_predictions,
                "quit_predictions": quit_predictions,
                "stay_predictions": stay_predictions,
                "quit_rate": quit_predictions / total_predictions if total_predictions > 0 else 0,
                "average_quit_probability": float(avg_quit_probability) if avg_quit_probability else 0,
                "confidence_distribution": confidence_stats,
                "generated_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Erreur statistiques: {e}")
        raise HTTPException(status_code=500, detail=str(e))