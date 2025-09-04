"""
Router pour les analytics et rapports basés sur les données PostgreSQL
Endpoints pour analyser les patterns d'attrition et les performances du modèle
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, case, cast, Integer

from app.database.connection import DatabaseManager
from app.database.models import Employee, PredictionSession, PredictionRequest, PredictionResult

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

@router.get("/attrition/overview")
async def get_attrition_overview(db_manager: DatabaseManager = Depends(get_database_manager)):
    """
    Vue d'ensemble de l'attrition dans le dataset original
    """
    try:
        with db_manager.get_session() as session:
            # Statistiques globales d'attrition
            total_employees = session.query(Employee).count()
            quit_employees = session.query(Employee).filter_by(a_quitte_l_entreprise="Oui").count()
            stay_employees = total_employees - quit_employees
            
            # Taux d'attrition par département
            dept_attrition = session.query(
                Employee.departement,
                func.count(Employee.employee_id).label('total'),
                func.sum(
                    case((Employee.a_quitte_l_entreprise == 'Oui', 1), else_=0)
                ).label('quit_count')
            ).group_by(Employee.departement).all()
            
            dept_stats = []
            for dept, total, quit_count in dept_attrition:
                quit_count = quit_count or 0
                dept_stats.append({
                    "department": dept,
                    "total_employees": total,
                    "quit_employees": quit_count,
                    "stay_employees": total - quit_count,
                    "attrition_rate": (quit_count / total) if total > 0 else 0
                })
            
            # Taux d'attrition par tranche d'âge
            age_groups = [
                (18, 30, "18-30"),
                (31, 40, "31-40"),
                (41, 50, "41-50"),
                (51, 65, "51+")
            ]
            
            age_stats = []
            for age_min, age_max, label in age_groups:
                if age_max == 65:
                    age_filter = Employee.age >= age_min
                else:
                    age_filter = and_(Employee.age >= age_min, Employee.age <= age_max)
                
                age_query = session.query(
                    func.count(Employee.employee_id).label('total'),
                    func.sum(
                        case((Employee.a_quitte_l_entreprise == 'Oui', 1), else_=0)
                    ).label('quit_count')
                ).filter(age_filter).first()
                
                total = age_query.total or 0
                quit_count = age_query.quit_count or 0
                
                age_stats.append({
                    "age_group": label,
                    "total_employees": total,
                    "quit_employees": quit_count,
                    "attrition_rate": (quit_count / total) if total > 0 else 0
                })
            
            # Moyennes de satisfaction pour les employés qui partent vs restent
            satisfaction_comparison = session.query(
                Employee.a_quitte_l_entreprise,
                func.avg(Employee.satisfaction_employee_environnement).label('avg_env'),
                func.avg(Employee.satisfaction_employee_nature_travail).label('avg_work'),
                func.avg(Employee.satisfaction_employee_equipe).label('avg_team'),
                func.avg(Employee.satisfaction_employee_equilibre_pro_perso).label('avg_balance')
            ).group_by(Employee.a_quitte_l_entreprise).all()
            
            satisfaction_stats = {}
            for status, avg_env, avg_work, avg_team, avg_balance in satisfaction_comparison:
                satisfaction_stats[status] = {
                    "avg_environment_satisfaction": float(avg_env) if avg_env else 0,
                    "avg_work_satisfaction": float(avg_work) if avg_work else 0,
                    "avg_team_satisfaction": float(avg_team) if avg_team else 0,
                    "avg_work_life_balance": float(avg_balance) if avg_balance else 0
                }
            
            return {
                "global_statistics": {
                    "total_employees": total_employees,
                    "quit_employees": quit_employees,
                    "stay_employees": stay_employees,
                    "overall_attrition_rate": quit_employees / total_employees if total_employees > 0 else 0
                },
                "by_department": dept_stats,
                "by_age_group": age_stats,
                "satisfaction_comparison": satisfaction_stats,
                "generated_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse d'attrition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/performance")
async def get_model_performance_analytics(
    days_back: int = Query(30, description="Période d'analyse en jours"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Analytics sur les performances du modèle de prédiction
    """
    try:
        with db_manager.get_session() as session:
            date_limit = datetime.now() - timedelta(days=days_back)
            
            # Statistiques globales des prédictions
            total_predictions = session.query(PredictionResult).filter(
                PredictionResult.created_at >= date_limit
            ).count()
            
            # Distribution des prédictions
            prediction_dist = session.query(
                PredictionResult.prediction,
                func.count(PredictionResult.result_id).label('count')
            ).filter(
                PredictionResult.created_at >= date_limit
            ).group_by(PredictionResult.prediction).all()
            
            prediction_distribution = {pred: count for pred, count in prediction_dist}
            
            # Distribution des niveaux de confiance
            confidence_dist = session.query(
                PredictionResult.confidence_level,
                func.count(PredictionResult.result_id).label('count'),
                func.avg(PredictionResult.probability_quit).label('avg_prob_quit')
            ).filter(
                PredictionResult.created_at >= date_limit
            ).group_by(PredictionResult.confidence_level).all()
            
            confidence_stats = []
            for conf_level, count, avg_prob in confidence_dist:
                confidence_stats.append({
                    "confidence_level": conf_level,
                    "prediction_count": count,
                    "avg_quit_probability": float(avg_prob) if avg_prob else 0
                })
            
            # Évolution temporelle (par jour)
            daily_predictions = session.query(
                func.date(PredictionResult.created_at).label('date'),
                func.count(PredictionResult.result_id).label('count'),
                func.avg(PredictionResult.probability_quit).label('avg_quit_prob'),
                func.sum(
                    case((PredictionResult.prediction == 'Oui', 1), else_=0)
                ).label('quit_predictions')
            ).filter(
                PredictionResult.created_at >= date_limit
            ).group_by(
                func.date(PredictionResult.created_at)
            ).order_by('date').all()
            
            daily_stats = []
            for date, count, avg_prob, quit_count in daily_predictions:
                daily_stats.append({
                    "date": date.isoformat(),
                    "total_predictions": count,
                    "quit_predictions": quit_count or 0,
                    "avg_quit_probability": float(avg_prob) if avg_prob else 0,
                    "predicted_attrition_rate": (quit_count or 0) / count if count > 0 else 0
                })
            
            # Top facteurs de risque
            risk_factors_query = session.query(PredictionResult.risk_factors).filter(
                PredictionResult.created_at >= date_limit,
                PredictionResult.risk_factors.isnot(None)
            ).all()
            
            # Comptage des facteurs de risque
            risk_factor_counts = {}
            for (risk_factors,) in risk_factors_query:
                if risk_factors:
                    for factor in risk_factors:
                        risk_factor_counts[factor] = risk_factor_counts.get(factor, 0) + 1
            
            # Top 10 des facteurs de risque
            top_risk_factors = sorted(
                risk_factor_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Temps de traitement moyen
            avg_processing_time = session.query(
                func.avg(PredictionResult.processing_time_ms)
            ).filter(
                PredictionResult.created_at >= date_limit,
                PredictionResult.processing_time_ms.isnot(None)
            ).scalar()
            
            return {
                "analysis_period": f"Last {days_back} days",
                "global_statistics": {
                    "total_predictions": total_predictions,
                    "prediction_distribution": prediction_distribution,
                    "avg_processing_time_ms": float(avg_processing_time) if avg_processing_time else None
                },
                "confidence_analysis": confidence_stats,
                "temporal_evolution": daily_stats,
                "top_risk_factors": [
                    {"factor": factor, "frequency": count} 
                    for factor, count in top_risk_factors
                ],
                "generated_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des performances: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/accuracy")
async def get_prediction_accuracy_analysis(
    days_back: int = Query(30, description="Période d'analyse en jours"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Analyse de précision du modèle (comparaison prédictions vs réalité si disponible)
    """
    try:
        with db_manager.get_session() as session:
            date_limit = datetime.now() - timedelta(days=days_back)
            
            # Analyse des prédictions pour les employés avec statut d'attrition connu
            accuracy_query = session.query(
                Employee.a_quitte_l_entreprise.label('actual'),
                PredictionResult.prediction.label('predicted'),
                PredictionResult.probability_quit,
                PredictionResult.confidence_level,
                func.count().label('count')
            ).join(
                PredictionRequest, Employee.employee_id == PredictionRequest.employee_id
            ).join(
                PredictionResult, PredictionRequest.request_id == PredictionResult.request_id
            ).filter(
                PredictionResult.created_at >= date_limit
            ).group_by(
                Employee.a_quitte_l_entreprise,
                PredictionResult.prediction,
                PredictionResult.probability_quit,
                PredictionResult.confidence_level
            ).all()
            
            # Matrice de confusion
            confusion_matrix = {
                "true_positives": 0,   # Prédit Oui, Réel Oui
                "true_negatives": 0,   # Prédit Non, Réel Non
                "false_positives": 0,  # Prédit Oui, Réel Non
                "false_negatives": 0   # Prédit Non, Réel Oui
            }
            
            detailed_results = []
            total_predictions = 0
            
            for actual, predicted, prob_quit, confidence, count in accuracy_query:
                total_predictions += count
                
                detailed_results.append({
                    "actual_attrition": actual,
                    "predicted_attrition": predicted,
                    "probability_quit": float(prob_quit),
                    "confidence_level": confidence,
                    "count": count,
                    "correct": actual == predicted
                })
                
                # Mise à jour de la matrice de confusion
                if actual == "Oui" and predicted == "Oui":
                    confusion_matrix["true_positives"] += count
                elif actual == "Non" and predicted == "Non":
                    confusion_matrix["true_negatives"] += count
                elif actual == "Non" and predicted == "Oui":
                    confusion_matrix["false_positives"] += count
                elif actual == "Oui" and predicted == "Non":
                    confusion_matrix["false_negatives"] += count
            
            # Calcul des métriques
            tp = confusion_matrix["true_positives"]
            tn = confusion_matrix["true_negatives"]
            fp = confusion_matrix["false_positives"]
            fn = confusion_matrix["false_negatives"]
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "analysis_period": f"Last {days_back} days",
                "total_analyzed_predictions": total_predictions,
                "confusion_matrix": confusion_matrix,
                "performance_metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                },
                "detailed_breakdown": detailed_results,
                "generated_at": datetime.now().isoformat(),
                "note": "Analyse basée sur les employés avec statut d'attrition connu"
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de précision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/employees/risk-segments")
async def get_employee_risk_segments(
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Segmentation des employés par niveau de risque d'attrition
    """
    try:
        with db_manager.get_session() as session:
            # Récupération des dernières prédictions par employé
            latest_predictions_subquery = session.query(
                PredictionRequest.employee_id,
                func.max(PredictionResult.created_at).label('latest_prediction')
            ).join(
                PredictionResult, PredictionRequest.request_id == PredictionResult.request_id
            ).filter(
                PredictionRequest.employee_id.isnot(None)
            ).group_by(PredictionRequest.employee_id).subquery()
            
            # Jointure pour récupérer les détails des dernières prédictions
            risk_analysis = session.query(
                Employee.employee_id,
                Employee.departement,
                Employee.age,
                Employee.poste,
                Employee.a_quitte_l_entreprise,
                PredictionResult.prediction,
                PredictionResult.probability_quit,
                PredictionResult.confidence_level,
                PredictionResult.risk_factors
            ).join(
                PredictionRequest, Employee.employee_id == PredictionRequest.employee_id
            ).join(
                PredictionResult, PredictionRequest.request_id == PredictionResult.request_id
            ).join(
                latest_predictions_subquery,
                and_(
                    Employee.employee_id == latest_predictions_subquery.c.employee_id,
                    PredictionResult.created_at == latest_predictions_subquery.c.latest_prediction
                )
            ).all()
            
            # Segmentation par niveau de risque
            risk_segments = {
                "high_risk": [],      # prob_quit >= 0.7
                "medium_risk": [],    # 0.3 <= prob_quit < 0.7
                "low_risk": []        # prob_quit < 0.3
            }
            
            for emp in risk_analysis:
                employee_data = {
                    "employee_id": emp.employee_id,
                    "department": emp.departement,
                    "age": emp.age,
                    "position": emp.poste,
                    "actual_attrition": emp.a_quitte_l_entreprise,
                    "predicted_attrition": emp.prediction,
                    "quit_probability": float(emp.probability_quit),
                    "confidence_level": emp.confidence_level,
                    "risk_factors": emp.risk_factors,
                    "prediction_accuracy": emp.a_quitte_l_entreprise == emp.prediction
                }
                
                if emp.probability_quit >= 0.7:
                    risk_segments["high_risk"].append(employee_data)
                elif emp.probability_quit >= 0.3:
                    risk_segments["medium_risk"].append(employee_data)
                else:
                    risk_segments["low_risk"].append(employee_data)
            
            # Statistiques par segment
            segment_stats = {}
            for segment_name, employees in risk_segments.items():
                if employees:
                    avg_prob = sum([emp["quit_probability"] for emp in employees]) / len(employees)
                    accuracy = sum([1 for emp in employees if emp["prediction_accuracy"]]) / len(employees)
                    dept_distribution = {}
                    
                    for emp in employees:
                        dept = emp["department"]
                        dept_distribution[dept] = dept_distribution.get(dept, 0) + 1
                    
                    segment_stats[segment_name] = {
                        "employee_count": len(employees),
                        "avg_quit_probability": avg_prob,
                        "prediction_accuracy": accuracy,
                        "department_distribution": dept_distribution
                    }
                else:
                    segment_stats[segment_name] = {
                        "employee_count": 0,
                        "avg_quit_probability": 0,
                        "prediction_accuracy": 0,
                        "department_distribution": {}
                    }
            
            return {
                "risk_segments": risk_segments,
                "segment_statistics": segment_stats,
                "total_analyzed_employees": len(risk_analysis),
                "generated_at": datetime.now().isoformat(),
                "note": "Basé sur les dernières prédictions disponibles par employé"
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la segmentation par risque: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends/department")
async def get_department_attrition_trends(
    days_back: int = Query(90, description="Période d'analyse en jours"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Tendances d'attrition par département basées sur les prédictions
    """
    try:
        with db_manager.get_session() as session:
            date_limit = datetime.now() - timedelta(days=days_back)
            
            # Analyse par département et par semaine
            weekly_dept_trends = session.query(
                func.extract('week', PredictionResult.created_at).label('week'),
                func.extract('year', PredictionResult.created_at).label('year'),
                Employee.departement,
                func.count(PredictionResult.result_id).label('total_predictions'),
                func.sum(
                    case((PredictionResult.prediction == 'Oui', 1), else_=0)
                ).label('quit_predictions'),
                func.avg(PredictionResult.probability_quit).label('avg_quit_probability')
            ).join(
                PredictionRequest, PredictionResult.request_id == PredictionRequest.request_id
            ).join(
                Employee, PredictionRequest.employee_id == Employee.employee_id
            ).filter(
                PredictionResult.created_at >= date_limit
            ).group_by(
                func.extract('week', PredictionResult.created_at),
                func.extract('year', PredictionResult.created_at),
                Employee.departement
            ).order_by('year', 'week', Employee.departement).all()
            
            # Organisation des données par département
            dept_trends = {}
            for week, year, dept, total, quit_pred, avg_prob in weekly_dept_trends:
                if dept not in dept_trends:
                    dept_trends[dept] = []
                
                quit_pred = quit_pred or 0
                dept_trends[dept].append({
                    "week": f"{int(year)}-W{int(week):02d}",
                    "total_predictions": total,
                    "quit_predictions": quit_pred,
                    "predicted_attrition_rate": quit_pred / total if total > 0 else 0,
                    "avg_quit_probability": float(avg_prob) if avg_prob else 0
                })
            
            # Statistiques globales par département
            dept_summary = session.query(
                Employee.departement,
                func.count(PredictionResult.result_id).label('total_predictions'),
                func.sum(
                    case((PredictionResult.prediction == 'Oui', 1), else_=0)
                ).label('quit_predictions'),
                func.avg(PredictionResult.probability_quit).label('avg_quit_probability'),
                func.stddev(PredictionResult.probability_quit).label('stddev_quit_probability')
            ).join(
                PredictionRequest, PredictionResult.request_id == PredictionRequest.request_id
            ).join(
                Employee, PredictionRequest.employee_id == Employee.employee_id
            ).filter(
                PredictionResult.created_at >= date_limit
            ).group_by(Employee.departement).all()
            
            summary_stats = []
            for dept, total, quit_pred, avg_prob, stddev_prob in dept_summary:
                quit_pred = quit_pred or 0
                summary_stats.append({
                    "department": dept,
                    "total_predictions": total,
                    "quit_predictions": quit_pred,
                    "predicted_attrition_rate": quit_pred / total if total > 0 else 0,
                    "avg_quit_probability": float(avg_prob) if avg_prob else 0,
                    "stddev_quit_probability": float(stddev_prob) if stddev_prob else 0
                })
            
            return {
                "analysis_period": f"Last {days_back} days",
                "weekly_trends_by_department": dept_trends,
                "department_summary": summary_stats,
                "generated_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des tendances départementales: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/predictions")
async def export_predictions_data(
    days_back: int = Query(30, description="Période d'export en jours"),
    format: str = Query("json", description="Format d'export (json uniquement pour l'instant)"),
    include_employee_data: bool = Query(False, description="Inclure les données complètes des employés"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    Export des données de prédiction pour analyse externe
    """
    try:
        with db_manager.get_session() as session:
            date_limit = datetime.now() - timedelta(days=days_back)
            
            if include_employee_data:
                # Export complet avec données employé
                export_query = session.query(
                    PredictionSession,
                    PredictionRequest,
                    PredictionResult,
                    Employee
                ).join(
                    PredictionRequest, PredictionSession.session_id == PredictionRequest.session_id
                ).join(
                    PredictionResult, PredictionRequest.request_id == PredictionResult.request_id
                ).outerjoin(
                    Employee, PredictionRequest.employee_id == Employee.employee_id
                ).filter(
                    PredictionResult.created_at >= date_limit
                ).order_by(PredictionResult.created_at).all()
                
                export_data = []
                for session_obj, request_obj, result_obj, employee_obj in export_query:
                    record = {
                        "session": session_obj.to_dict(),
                        "request": request_obj.to_dict(),
                        "result": result_obj.to_dict(),
                        "employee": employee_obj.to_dict() if employee_obj else None
                    }
                    export_data.append(record)
            
            else:
                # Export simplifié prédictions seulement
                export_query = session.query(
                    PredictionSession.session_id,
                    PredictionSession.session_type,
                    PredictionSession.started_at,
                    PredictionRequest.request_id,
                    PredictionRequest.employee_id,
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
                ).order_by(PredictionResult.created_at).all()
                
                export_data = []
                for row in export_query:
                    record = {
                        "session_id": str(row.session_id),
                        "session_type": row.session_type,
                        "session_started": row.started_at.isoformat(),
                        "request_id": row.request_id,
                        "employee_id": row.employee_id,
                        "prediction": row.prediction,
                        "probability_quit": float(row.probability_quit),
                        "confidence_level": row.confidence_level,
                        "risk_factors": row.risk_factors,
                        "model_version": row.model_version,
                        "predicted_at": row.created_at.isoformat()
                    }
                    export_data.append(record)
            
            return {
                "export_metadata": {
                    "period": f"Last {days_back} days",
                    "export_date": datetime.now().isoformat(),
                    "format": format,
                    "include_employee_data": include_employee_data,
                    "total_records": len(export_data)
                },
                "data": export_data
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de l'export des données: {e}")
        raise HTTPException(status_code=500, detail=str(e))