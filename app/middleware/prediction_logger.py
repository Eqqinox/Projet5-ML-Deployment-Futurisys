"""
Middleware de traçabilité automatique pour les prédictions ML
Enregistre automatiquement tous les appels API en base de données
"""

import time
import json
import logging
import uuid
from typing import Callable
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.database.models import APIAuditLog

logger = logging.getLogger(__name__)

class PredictionLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware pour tracer automatiquement tous les appels API
    et créer des sessions de prédiction pour les endpoints ML
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.prediction_endpoints = {
            "/api/v1/predict/single",
            "/api/v1/predict/batch",
            "/api/v1/predict/validate-input"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Traite chaque requête HTTP et enregistre les métadonnées
        """
        start_time = time.time()
        
        # Extraction des métadonnées de la requête
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        endpoint_path = str(request.url.path)
        http_method = request.method
        
        # Headers de la requête (filtrage des données sensibles)
        request_headers = dict(request.headers)
        filtered_headers = self._filter_sensitive_headers(request_headers)
        
        # Lecture du payload de la requête
        request_payload = None
        if http_method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    request_payload = json.loads(body.decode())
                    # Créer une nouvelle requête avec le même body
                    request = self._recreate_request(request, body)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Impossible de parser le payload de la requête: {e}")
                request_payload = {"error": "unparseable_payload"}
        
        # Gestion spéciale pour les endpoints de prédiction
        session_id = None
        if endpoint_path in self.prediction_endpoints:
            session_id = self._create_or_get_session(request, request_payload)
            # Ajouter l'ID de session aux headers pour les endpoints
            if hasattr(request, 'state'):
                request.state.prediction_session_id = session_id
        
        # Traitement de la requête
        try:
            response = await call_next(request)
            
            # Calcul du temps de traitement
            processing_time = (time.time() - start_time) * 1000  # en ms
            
            # Lecture du payload de la réponse
            response_payload = None
            if hasattr(response, 'body'):
                try:
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk
                    
                    if response_body:
                        response_payload = json.loads(response_body.decode())
                    
                    # Recréer la réponse avec le même contenu
                    response = Response(
                        content=response_body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                    
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    response_payload = {"info": "non_json_response"}
            
            # Enregistrement en base de données (asynchrone)
            await self._log_api_call(
                session_id=session_id,
                endpoint=endpoint_path,
                http_method=http_method,
                client_ip=client_ip,
                user_agent=user_agent,
                request_headers=filtered_headers,
                request_payload=request_payload,
                response_status_code=response.status_code,
                response_payload=response_payload,
                response_time_ms=processing_time,
                request=request
            )
            
            return response
            
        except Exception as e:
            # En cas d'erreur, logger quand même l'appel
            processing_time = (time.time() - start_time) * 1000
            
            error_response = {
                "error": "internal_server_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            await self._log_api_call(
                session_id=session_id,
                endpoint=endpoint_path,
                http_method=http_method,
                client_ip=client_ip,
                user_agent=user_agent,
                request_headers=filtered_headers,
                request_payload=request_payload,
                response_status_code=500,
                response_payload=error_response,
                response_time_ms=processing_time,
                request=request
            )
            
            # Relancer l'exception
            raise e
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extrait l'adresse IP du client en tenant compte des proxies
        """
        # Vérifier les headers de proxy
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # IP directe
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _filter_sensitive_headers(self, headers: dict) -> dict:
        """
        Filtre les headers sensibles (auth, cookies, etc.)
        """
        sensitive_headers = {
            "authorization", "cookie", "x-api-key", 
            "x-auth-token", "x-csrf-token"
        }
        
        filtered = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                filtered[key] = "***FILTERED***"
            else:
                filtered[key] = value
        
        return filtered
    
    def _recreate_request(self, original_request: Request, body: bytes) -> Request:
        """
        Recrée une requête avec le même body (FastAPI consomme le stream)
        """
        from starlette.requests import Request as StarletteRequest
        
        # Créer un nouveau scope avec le body
        scope = original_request.scope.copy()
        
        # Simuler le stream du body
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        
        return Request(scope, receive)
    
    def _create_or_get_session(self, request: Request, payload: dict) -> str:
        """
        Crée ou récupère une session de prédiction
        """
        try:
            # Vérifier si une session existe déjà dans les headers
            existing_session = request.headers.get("x-prediction-session-id")
            if existing_session:
                return existing_session
            
            # Déterminer le type de session basé sur l'endpoint
            session_type = "single"
            if "batch" in str(request.url.path):
                session_type = "batch"
            
            # Métadonnées de la session
            session_metadata = {
                "endpoint": str(request.url.path),
                "user_agent": request.headers.get("user-agent", ""),
                "client_ip": self._get_client_ip(request),
                "timestamp": datetime.now().isoformat()
            }
            
            # Créer la session via le DatabaseManager si disponible
            if hasattr(request.app.state, 'db_manager'):
                db_manager = request.app.state.db_manager
                session_id = db_manager.create_prediction_session(
                    session_type=session_type,
                    metadata=session_metadata
                )
                return session_id
            else:
                # Fallback: générer un UUID
                return str(uuid.uuid4())
                
        except Exception as e:
            logger.error(f"Erreur lors de la création de session: {e}")
            return str(uuid.uuid4())
    
    async def _log_api_call(
        self, 
        session_id: str,
        endpoint: str,
        http_method: str,
        client_ip: str,
        user_agent: str,
        request_headers: dict,
        request_payload: dict,
        response_status_code: int,
        response_payload: dict,
        response_time_ms: float,
        request: Request
    ):
        """
        Enregistre l'appel API en base de données
        """
        try:
            # Vérifier si le DatabaseManager est disponible
            if not hasattr(request.app.state, 'db_manager'):
                logger.warning("DatabaseManager non disponible pour l'audit")
                return
            
            db_manager = request.app.state.db_manager
            
            # Utiliser une session de base de données
            with db_manager.get_session() as db_session:
                audit_log = APIAuditLog(
                    session_id=uuid.UUID(session_id) if session_id else None,
                    endpoint_called=endpoint,
                    http_method=http_method,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    request_headers=request_headers,
                    request_payload=request_payload,
                    response_status_code=response_status_code,
                    response_payload=response_payload,
                    response_time_ms=response_time_ms
                )
                
                db_session.add(audit_log)
                # Le commit est géré par le context manager
                
                logger.debug(f"Appel API enregistré: {http_method} {endpoint} ({response_status_code})")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de l'audit: {e}")
            # Ne pas faire crash l'application pour un problème d'audit


class PredictionSessionManager:
    """
    Gestionnaire de sessions de prédiction pour les endpoints ML
    Utilisé par les routers de prédiction pour gérer la traçabilité
    """
    
    @staticmethod
    def get_session_id_from_request(request: Request) -> str:
        """
        Récupère l'ID de session depuis la requête
        """
        # Priorité aux headers
        session_id = request.headers.get("x-prediction-session-id")
        if session_id:
            return session_id
        
        # Fallback sur l'état de la requête (défini par le middleware)
        if hasattr(request, 'state') and hasattr(request.state, 'prediction_session_id'):
            return request.state.prediction_session_id
        
        # Créer une nouvelle session si aucune n'existe
        return str(uuid.uuid4())
    
    @staticmethod
    async def save_prediction_input(
        request: Request, 
        input_data: dict, 
        employee_id: int = None
    ) -> int:
        """
        Sauvegarde un input de prédiction
        """
        if not hasattr(request.app.state, 'db_manager'):
            logger.warning("DatabaseManager non disponible pour sauvegarder l'input")
            return None
        
        try:
            db_manager = request.app.state.db_manager
            session_id = PredictionSessionManager.get_session_id_from_request(request)
            
            request_id = db_manager.save_prediction_request(
                session_id=session_id,
                input_data=input_data,
                employee_id=employee_id
            )
            
            logger.debug(f"Input de prédiction sauvegardé: request_id={request_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'input: {e}")
            return None
    
    @staticmethod
    async def save_prediction_output(request: Request, request_id: int, prediction_result) -> int:
        """
        Sauvegarde un output de prédiction
        """
        if not hasattr(request.app.state, 'db_manager'):
            logger.warning("DatabaseManager non disponible pour sauvegarder l'output")
            return None
        
        try:
            db_manager = request.app.state.db_manager
            
            result_id = db_manager.save_prediction_result(
                request_id=request_id,
                prediction_result=prediction_result
            )
            
            logger.debug(f"Output de prédiction sauvegardé: result_id={result_id}")
            return result_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'output: {e}")
            return None
    
    @staticmethod
    async def complete_session(request: Request, total_predictions: int):
        """
        Marque une session comme terminée
        """
        if not hasattr(request.app.state, 'db_manager'):
            return
        
        try:
            db_manager = request.app.state.db_manager
            session_id = PredictionSessionManager.get_session_id_from_request(request)
            
            db_manager.complete_prediction_session(
                session_id=session_id,
                total_predictions=total_predictions
            )
            
            logger.debug(f"Session terminée: {session_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la completion de session: {e}")