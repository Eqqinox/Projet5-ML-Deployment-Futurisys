"""
Router pour les endpoints de santé et de monitoring
Vérification de l'état de l'API
"""
from fastapi import APIRouter
from typing import Dict
from datetime import datetime

router = APIRouter()

@router.get("/", response_model=Dict)
async def health_check():
    """
    Vérification de santé basique de l'API
    """
    return {
        "status": "healthy",
        "service": "Futurisys ML API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
