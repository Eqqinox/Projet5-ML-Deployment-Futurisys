# Standards de Code et Expérimentation ML - Projet Futurisys

## Vue d'ensemble

Ce document établit les conventions de développement et les bonnes pratiques d'expérimentation ML appliquées dans le projet Futurisys. Ces standards garantissent la reproductibilité, la maintenabilité et la qualité du code en production.

## Structure de Projet Standardisée

### Architecture
```
Projet5/
├── app/                          # Code application (PEP8 strict)
│   ├── core/                     # Configuration centralisée
│   ├── models/                   # Modèles ML + schémas Pydantic
│   ├── routers/                  # Endpoints FastAPI modulaires
│   ├── database/                 # ORM SQLAlchemy + connexions
│   ├── middleware/               # Middleware personnalisés
│   └── utils/                    # Fonctions utilitaires réutilisables
├── tests/                        # Tests automatisés (pytest)
├── docs/                         # Documentation technique
├── data/                         # Données (raw/processed/external)
├── database/                     # Scripts BDD + documentation
└── requirements.txt              # Dépendances figées avec versions
```

## Standards de Code Python

### Conventions de nommage (PEP8)

**Variables et fonctions** : snake_case
```python
# ✅ Correct
employee_data = request.json()
def predict_single_employee(data):
    return model.predict(data)

# ❌ Incorrect
employeeData = request.json()
def predictSingleEmployee(data):
    return model.predict(data)
```

**Classes** : PascalCase
```python
# ✅ Correct
class MLModel:
    pass

class PredictionResult:
    pass

# ❌ Incorrect
class ml_model:
    pass
```

**Constantes** : UPPER_SNAKE_CASE
```python
# ✅ Correct
MODEL_VERSION = "1.0.0"
DEFAULT_THRESHOLD = 0.514
MAX_BATCH_SIZE = 100
```

### Documentation et commentaires

**Docstrings obligatoires** pour toutes les fonctions publiques
```python
def predict_single(self, employee: EmployeeData) -> PredictionResult:
    """
    Effectue une prédiction pour un seul employé
    
    Args:
        employee (EmployeeData): Données validées de l'employé
        
    Returns:
        PredictionResult: Prédiction avec probabilités et facteurs de risque
        
    Raises:
        ValueError: Si les données ne passent pas la validation
        RuntimeError: Si le modèle n'est pas chargé
    """
```

**Commentaires techniques** pour la logique ML complexe
```python
# Encodage OneHot reproductible avec encodeurs pré-entraînés (Projet 4)
df_categorical = df[self.variables_catego]
df_encoded = self.onehot_encoder.transform(df_categorical)

# Seuil optimisé par validation croisée (F1-Score maximal)
prediction = "Oui" if prob_quit >= self.threshold else "Non"
```

### Gestion d'erreurs

**Exceptions spécifiques** avec messages informatifs
```python
# ✅ Correct - Erreurs spécifiques
try:
    self.model = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
except Exception as e:
    logger.error(f"Erreur chargement modèle: {e}")
    raise RuntimeError(f"Impossible de charger le modèle: {e}")

# ❌ Incorrect - Trop générique
try:
    self.model = joblib.load(model_path)
except Exception:
    print("Erreur modèle")
```

**Logging structuré** pour traçabilité
```python
import logging

logger = logging.getLogger(__name__)

def predict_batch(self, employees):
    logger.info(f"Prédiction batch pour {len(employees)} employés")
    
    try:
        predictions = []
        for i, employee in enumerate(employees):
            prediction = self.predict_single(employee)
            predictions.append(prediction)
            
        logger.info(f"Batch terminé: {len(predictions)} prédictions réussies")
        return predictions
        
    except Exception as e:
        logger.error(f"Échec prédiction batch: {e}")
        raise
```

## Standards d'Expérimentation ML

### Reproductibilité obligatoire

**Seeds fixes** pour tous les composants aléatoires
```python
# Configuration globale de reproductibilité
RANDOM_STATE = 42

# XGBoost
model = XGBClassifier(
    random_state=RANDOM_STATE,
    n_estimators=200,
    # ... autres hyperparamètres
)

# Scikit-learn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

**Versions figées** des dépendances critiques
```python
# requirements.txt - Versions exactes pour ML
numpy==1.26.0              # Version fixée pour reproductibilité XGBoost
xgboost>=3.0.4,<4.0.0      # Plage compatible
scikit-learn>=1.7.1,<2.0.0 # Compatibilité encodeurs
pandas>=2.3.1,<3.0.0       # API stable
```

### Validation et métriques standardisées

**Validation croisée stratifiée** (obligatoire pour déséquilibre)
```python
from sklearn.model_selection import StratifiedKFold, cross_validate

# Configuration validation standard
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Métriques standardisées pour classification binaire
scoring_metrics = [
    'accuracy',      # Précision globale
    'precision',     # Précision positive (éviter faux positifs)
    'recall',        # Rappel (détecter vrais positifs)
    'f1',           # Équilibre précision/rappel
    'roc_auc'       # Capacité discrimination
]

# Validation complète avec écarts-types
cv_results = cross_validate(
    model, X_train_processed, y_train,
    cv=cv_strategy,
    scoring=scoring_metrics,
    return_train_score=False
)

# Reporting standardisé
for metric in scoring_metrics:
    mean_score = cv_results[f'test_{metric}'].mean()
    std_score = cv_results[f'test_{metric}'].std()
    print(f"{metric.title()}: {mean_score:.4f} (± {std_score:.4f})")
```

### Optimisation des hyperparamètres

**Stratégie systématique** pour éviter l'overfitting
```python
# 1. Baseline rapide avec paramètres par défaut
baseline_model = XGBClassifier(random_state=42)
baseline_score = cross_val_score(baseline_model, X_train, y_train, cv=5).mean()

# 2. Grid Search ciblé sur hyperparamètres critiques
param_grid = {
    'n_estimators': [100, 200, 300],      # Nombre d'arbres
    'max_depth': [3, 4, 5],               # Profondeur (overfitting)
    'learning_rate': [0.05, 0.1, 0.15],   # Vitesse apprentissage
    'subsample': [0.6, 0.8, 1.0],         # Échantillonnage lignes
    'scale_pos_weight': [4, 5, 6]         # Déséquilibre classes
}

# 3. Optimisation avec validation croisée interne
grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1',  # Métrique d'optimisation
    n_jobs=-1
)
```

### Sérialisation et versioning des modèles

**Sauvegarde complète** des artefacts ML
```python
import joblib
from pathlib import Path

def save_model_artifacts(model, encoders, metadata, version="1.0.0"):
    """
    Sauvegarde complète des artefacts ML pour reproductibilité
    """
    models_dir = Path("app/models")
    models_dir.mkdir(exist_ok=True)
    
    # Modèle principal
    joblib.dump(model, models_dir / "trained_model.pkl")
    
    # Encodeurs (preprocessing reproductible)
    joblib.dump(encoders['onehot'], models_dir / "onehot_encoder.pkl")
    joblib.dump(encoders['ordinal'], models_dir / "ordinal_encoder.pkl")
    
    # Ordre des features après encodage
    joblib.dump(encoders['column_names'], models_dir / "final_column_names.pkl")
    
    # Métadonnées complètes
    model_info = {
        "version": version,
        "algorithm": "XGBoost",
        "threshold_optimized": 0.514,
        "cv_results": metadata.get("cv_results", {}),
        "feature_importance": dict(zip(
            encoders['column_names'], 
            model.feature_importances_
        )),
        "hyperparameters": model.get_params(),
        "training_timestamp": datetime.now().isoformat()
    }
    
    joblib.dump(model_info, models_dir / "model_info.pkl")
    
    print(f"✅ Modèle v{version} sauvegardé avec tous les artefacts")
```

## Standards FastAPI et API

### Structure des endpoints

**Organisation modulaire** par fonctionnalité
```python
# app/routers/predictions.py
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/api/v1/predict", tags=["Predictions"])

@router.post("/single", response_model=PredictionResult)
async def predict_single_employee(
    employee: EmployeeData,
    ml_model: MLModel = Depends(get_ml_model)
):
    """Documentation OpenAPI complète avec exemples"""
```

**Validation Pydantic** stricte avec contraintes métier
```python
from pydantic import BaseModel, Field, validator

class EmployeeData(BaseModel):
    satisfaction_employee_environnement: int = Field(
        ..., ge=1, le=4,
        description="Satisfaction environnement de travail (1-4)"
    )
    
    age: int = Field(
        ..., ge=18, le=65,
        description="Âge de l'employé"
    )
    
    @validator('augementation_salaire_precedente')
    def validate_salary_increase(cls, v):
        if not (0.11 <= v <= 0.25):
            raise ValueError('Augmentation doit être entre 11% et 25%')
        return v
```

### Gestion des dépendances

**Injection de dépendances** pour testabilité
```python
# app/main.py - Configuration globale
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement modèle au démarrage (une seule fois)
    ml_model = MLModel()
    ml_model.load_model()
    app.state.ml_model = ml_model
    
    yield
    
    # Nettoyage
    app.state.ml_model = None

# Dependency injection
def get_ml_model() -> MLModel:
    from app.main import app
    if not hasattr(app.state, 'ml_model') or app.state.ml_model is None:
        raise HTTPException(503, "Modèle ML non disponible")
    return app.state.ml_model
```

## Standards de Tests

### Architecture de tests (85 tests implémentés)

**Tests unitaires** (test_api_endpoints.py) - 34 tests
```python
import pytest
from fastapi.testclient import TestClient

class TestPredictionEndpoints:
    """Tests unitaires des endpoints de prédiction"""
    
    def test_predict_single_valid_input(self, client: TestClient):
        """Test prédiction avec données valides"""
        valid_employee = {
            "satisfaction_employee_environnement": 4,
            "age": 35,
            # ... autres champs requis
        }
        
        response = client.post("/api/v1/predict/single", json=valid_employee)
        
        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result
        assert result["prediction"] in ["Oui", "Non"]
        assert 0 <= result["probability_quit"] <= 1
```

**Tests d'intégration** (test_integration.py) - 19 tests
```python
def test_full_prediction_workflow_with_database(client, db_session):
    """Test workflow complet avec traçabilité PostgreSQL"""
    # 1. Prédiction
    response = client.post("/api/v1/predict/single", json=valid_employee)
    
    # 2. Vérification base de données
    session_count = db_session.query(PredictionSession).count()
    assert session_count > 0
    
    # 3. Cohérence données
    result = response.json()
    db_result = db_session.query(PredictionResult).first()
    assert db_result.prediction == result["prediction"]
```

**Tests de performance** (test_performance.py) - 17 tests
```python
import time

def test_prediction_response_time():
    """Temps de réponse < 200ms pour prédiction individuelle"""
    start_time = time.time()
    
    response = client.post("/api/v1/predict/single", json=valid_employee)
    
    response_time = (time.time() - start_time) * 1000  # ms
    assert response_time < 200, f"Trop lent: {response_time:.1f}ms"
```

**Tests cas limites** (test_edge_cases.py) - 15 tests
```python
def test_employee_high_risk_prediction():
    """Test employé à haut risque avec données réelles Projet 4"""
    high_risk_employee = {
        "satisfaction_employee_environnement": 1,  # Très insatisfait
        "heure_supplementaires": "Oui",            # Surcharge
        "augementation_salaire_precedente": 0.11,  # Augmentation minimale
        # ... profil complet à risque
    }
    
    response = client.post("/api/v1/predict/single", json=high_risk_employee)
    result = response.json()
    
    # Vérifications métier
    assert result["prediction"] == "Oui"
    assert result["probability_quit"] > 0.7  # Haute probabilité
    assert len(result["risk_factors"]) > 0   # Facteurs identifiés
```

### Couverture de code et métriques

**Configuration pytest.ini**
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = 
    --cov=app
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=50
markers =
    integration: Tests d'intégration (nécessitent PostgreSQL)
    performance: Tests de performance et charge
```

## Standards de Documentation

### Documentation technique obligatoire

**README principal** : Vue d'ensemble complète
- Architecture et stack technique
- Instructions d'installation détaillées
- Exemples d'utilisation avec code
- Métriques de performance validées

**Documentation API** : Auto-générée FastAPI/OpenAPI
- Descriptions détaillées des endpoints
- Schémas de validation avec exemples
- Codes d'erreur et gestion d'exceptions
- Guide d'intégration pour développeurs

**Documentation ML** : `docs/model_documentation.md`
- Architecture et hyperparamètres du modèle
- Métriques de performance avec validation croisée
- Pipeline de preprocessing détaillé
- Explicabilité et facteurs de risque

**Guide de maintenance** : `docs/maintenance_guide.md`
- Procédures de monitoring et health checks
- Scripts de mise à jour et rollback
- Gestion des versions et déploiement
- Archivage et nettoyage des données

## Standards de Configuration

### Gestion des environnements

**Configuration centralisée** avec Pydantic Settings
```python
# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration avec validation automatique des types"""
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Base de données
    DATABASE_URL: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    
    # ML Model
    MODEL_PATH: str = "app/models/trained_model.pkl"
    MODEL_VERSION: str = "1.0.0"
    MAX_BATCH_SIZE: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

**Fichiers d'environnement** structurés
```bash
# .env (développement local)
DATABASE_URL=postgresql://user:pass@localhost:5432/futurisys_ml
DEBUG=True
LOG_LEVEL=DEBUG

# .env.production (déploiement)
DATABASE_URL=postgresql://prod_user:secure_pass@prod_host:5432/futurisys_ml
DEBUG=False
LOG_LEVEL=INFO
SECRET_KEY=production-secret-key-very-secure
```

### Logging standardisé

**Configuration centralisée** des logs
```python
# app/core/config.py - LOGGING_CONFIG
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "detailed"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": settings.LOG_FILE,
            "level": settings.LOG_LEVEL,
            "formatter": "detailed"
        }
    },
    "loggers": {
        "app": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
}
```

## Standards DevOps et Déploiement

### Pipeline CI/CD (GitHub Actions)

**Tests automatisés** avant déploiement
```yaml
# .github/workflows/deploy-hf-spaces.yml
name: Deploy to Hugging Face Spaces

on:
  push:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Test FastAPI app loads
      run: |
        python -c "
        import sys
        sys.path.insert(0, '.')
        from app.main import app
        print('✅ FastAPI app loads successfully')
        "
```

**Déploiement conditionnel** par branche
```yaml
  deploy-dev:
    needs: test
    if: github.ref == 'refs/heads/develop'
    # ... déploiement environnement développement
  
  deploy-prod:
    needs: test  
    if: github.ref == 'refs/heads/main'
    # ... déploiement production
```

### Containerisation Docker

**Multi-stage build** optimisé
```dockerfile
# Dockerfile
FROM python:3.11-slim as base

# Variables d'environnement
ENV PYTHONPATH=/code
ENV PORT=7860

# Dépendances système
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code application
COPY . /code/

# Point d'entrée Hugging Face Spaces
CMD ["python", "hf_app.py"]
```

## Standards de Sécurité

### Validation d'entrée stricte

**Sanitization** des données utilisateur
```python
from pydantic import validator, Field
import re

class EmployeeData(BaseModel):
    # Validation stricte avec regex
    genre: str = Field(..., regex=r'^(Homme|Femme))
    
    @validator('revenu_mensuel')
    def validate_salary_range(cls, v):
        if not (1000 <= v <= 50000):
            raise ValueError('Revenu doit être réaliste (1000-50000€)')
        return v
    
    @validator('*', pre=True)
    def strip_whitespace(cls, v):
        # Suppression espaces superflus pour tous les champs string
        return v.strip() if isinstance(v, str) else v
```

**Headers HTTP** filtrés pour audit
```python
def _filter_sensitive_headers(self, headers: dict) -> dict:
    """Filtrage headers sensibles pour logs audit"""
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
```

## Standards de Performance

### Monitoring et métriques

**Health checks** multicouches
```python
@router.get("/health/")
async def health_check():
    """Vérification santé API"""
    return {
        "status": "healthy",
        "service": "Futurisys ML API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/health/detailed")
async def detailed_health_check(ml_model: MLModel = Depends(get_ml_model)):
    """Vérification détaillée tous composants"""
    health_status = {
        "api": True,
        "model_loaded": ml_model.is_loaded,
        "model_file_exists": ml_model.health_check()["model_file_exists"],
        "database": check_database_connection(),
        "encoders": ml_model.health_check()["encoders_loaded"]
    }
    
    overall_status = "healthy" if all(health_status.values()) else "degraded"
    
    return {
        "status": overall_status,
        "components": health_status,
        "timestamp": datetime.now().isoformat()
    }
```

**Temps de réponse** surveillés
```python
import time
from functools import wraps

def monitor_performance(func):
    """Décorateur monitoring temps d'exécution"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"{func.__name__} executed in {processing_time:.1f}ms")
            
            return result
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"{func.__name__} failed after {processing_time:.1f}ms: {e}")
            raise
    
    return wrapper
```

## Bonnes Pratiques Spécifiques ML

### Drift Detection et Monitoring

**Baseline de référence** pour détecter la dérive
```python
def check_model_drift(new_predictions, reference_metrics):
    """
    Détection simple de dérive modèle
    """
    current_positive_rate = sum(1 for p in new_predictions if p == "Oui") / len(new_predictions)
    baseline_positive_rate = reference_metrics.get("baseline_positive_rate", 0.16)
    
    # Alerte si dérive > 5% du baseline
    drift_threshold = 0.05
    if abs(current_positive_rate - baseline_positive_rate) > drift_threshold:
        logger.warning(
            f"Dérive détectée: taux positif {current_positive_rate:.2%} "
            f"vs baseline {baseline_positive_rate:.2%}"
        )
        return True
    
    return False
```

### Explicabilité et audit trail

**Traçabilité complète** des décisions ML
```python
class PredictionAuditTrail:
    """Audit trail complet pour prédictions ML"""
    
    @staticmethod
    def log_prediction(input_data, output_data, model_metadata):
        """Enregistrement complet prédiction pour audit"""
        audit_record = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_metadata["version"],
            "model_threshold": model_metadata["threshold"],
            "input_hash": hashlib.sha256(str(input_data).encode()).hexdigest()[:16],
            "prediction": output_data["prediction"],
            "confidence": output_data["confidence_level"],
            "risk_factors": output_data["risk_factors"],
            "processing_time_ms": output_data.get("processing_time_ms")
        }
        
        # Stockage sécurisé pour conformité réglementaire
        store_audit_record(audit_record)
```

---

## Mise en Application

### Checklist avant mise en production

**Code Quality**
- [ ] PEP8 respecté (flake8 / black)
- [ ] Docstrings complètes
- [ ] Type hints sur fonctions publiques
- [ ] Gestion d'erreurs robuste
- [ ] Logs structurés implémentés

**Testing**
- [ ] Tests unitaires >90% couverture modules critiques
- [ ] Tests d'intégration avec base de données
- [ ] Tests de performance validés (<200ms)
- [ ] Tests cas limites métier

**Documentation**
- [ ] README à jour avec exemples
- [ ] Documentation API complète
- [ ] Guide de maintenance rédigé
- [ ] Architecture documentée

**Sécurité & Performance**
- [ ] Validation stricte inputs
- [ ] Health checks implémentés
- [ ] Monitoring logs configuré
- [ ] Variables d'environnement sécurisées

### Processus de review

**Pull Request** obligatoire avec template
```markdown
## Description
Brief description des changements

## Type de changement
- [ ] Bug fix
- [ ] Nouvelle fonctionnalité  
- [ ] Breaking change
- [ ] Documentation

## Tests
- [ ] Tests existants passent
- [ ] Nouveaux tests ajoutés si nécessaire
- [ ] Tests manuels effectués

## Checklist
- [ ] Code suit les standards PEP8
- [ ] Documentation mise à jour
- [ ] Pas de hardcoding de secrets
```

---

**Document maintenu par** : MMEKNACI 
**Dernière révision** : Septembre 2025  
**Version standards** : 1.0.0 (Projet étudiant OpenClassrooms - Data Scientist et Machine Learning)