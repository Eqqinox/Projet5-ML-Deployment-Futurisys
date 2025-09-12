"""
Router pour les endpoints de santé et de monitoring
Vérification de l'état de l'API
"""
from fastapi import APIRouter
from typing import Dict
from datetime import datetime

router = APIRouter()

@router.get(
    "/",
    response_model=Dict,
    summary="Vérification de santé de l'API",
    description="""
    Endpoint de monitoring principal pour vérifier l'état de l'API Futurisys.
    
    Retourne le statut de santé, la version du service et un timestamp.
    Utilisé par les systèmes de monitoring automatique.
    """,
    responses={
        200: {
            "description": "API fonctionnelle",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "Futurisys ML API",
                        "version": "1.0.0",
                        "timestamp": "2025-09-11T14:30:00.123456"
                    }
                }
            }
        }
    }
)
async def health_check():
    """Vérification de santé basique de l'API"""
    return {
        "status": "healthy",
        "service": "Futurisys ML API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
