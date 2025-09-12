# Architecture Overview - Futurisys ML API

## 1. Vue d'ensemble système

### Architecture globale
L'API Futurisys ML suit une **architecture en couches** basée sur FastAPI avec séparation des responsabilités :

```
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE PRÉSENTATION                      │
├─────────────────────────────────────────────────────────────┤
│  Client HTTP → FastAPI → Middleware → Routers → Responses   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   COUCHE LOGIQUE MÉTIER                     │
├─────────────────────────────────────────────────────────────┤
│  MLModel (XGBoost) → Preprocessing → Prédictions → Results  │
│  PredictionLogger → Sessions → Traçabilité → Audit          │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE PERSISTANCE                       │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL → SQLAlchemy → Modèles → Stockage → Historique  │
└─────────────────────────────────────────────────────────────┘
```

### Stack technique déployée
- **Framework API** : FastAPI 0.104+ (async/await natif)
- **Serveur ASGI** : Uvicorn (production-ready)
- **Modèle ML** : XGBoost 3.0.4 (gradient boosting optimisé)
- **Base de données** : PostgreSQL 13+ (mode local pour ce projet)
- **ORM** : SQLAlchemy 2.0+ (relations et migrations)
- **Validation** : Pydantic 2.5+ (type hints et validation automatique)
- **Déploiement** : Docker → Hugging Face Spaces
- **CI/CD** : GitHub Actions (tests automatisés + déploiement)

## 2. Flux de données détaillé

### Parcours d'une requête de prédiction complète

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Middleware
    participant MLModel
    participant PostgreSQL
    
    Client->>FastAPI: POST /api/v1/predict/single
                  ou: POST /api/v1/predict/batch
    Note over Client,FastAPI: Données employé (27 variables)
    
    FastAPI->>Middleware: PredictionLoggerMiddleware
    Note over Middleware: Création session UUID + capture requête
    
    Middleware->>PostgreSQL: Sauvegarde session + input
    Note over PostgreSQL: Tables: prediction_sessions, prediction_requests
    
    FastAPI->>MLModel: predict_single(employee_data)
    Note over MLModel: Preprocessing + XGBoost + Postprocessing
    
    MLModel->>MLModel: 1. Encodage OneHot/Ordinal
    MLModel->>MLModel: 2. Prédiction XGBoost (seuil 0.514)
    MLModel->>MLModel: 3. Calcul probabilités + facteurs risque
    
    MLModel->>FastAPI: PredictionResult (JSON)
    
    FastAPI->>Middleware: Réponse + métadonnées
    Middleware->>PostgreSQL: Sauvegarde résultat
    Note over PostgreSQL: Table: prediction_results
    
    FastAPI->>Client: Réponse JSON complète
    Note over Client,FastAPI: Prédiction + probabilités + explicabilité
```

### Ordre de traitement des données ML

1. **Réception** : Validation Pydantic (27 variables employé)
2. **Preprocessing** : 
   - Encodage binaire : `heure_supplementaires` (Oui/Non → 1/0), `genre` (Homme/Femme → 1/0)
   - Encodage ordinal : `frequence_deplacement` (3 niveaux)
   - Encodage OneHot : 4 variables catégorielles (`departement`, `poste`, `statut_marital`, `domaine_etude`)
   - Réorganisation : 40 features finales selon ordre d'entraînement
3. **Prédiction** : XGBoost avec seuil optimisé (0.514)
4. **Post-processing** : Calcul probabilités, niveau confiance, facteurs risque
5. **Réponse** : Sérialisation JSON via Pydantic

### Gestion de la traçabilité (interceptée par middleware)

- **AVANT prédiction** : Sauvegarde input dans `prediction_requests`
- **APRÈS prédiction** : Sauvegarde output dans `prediction_results`
- **Mode synchrone** : Chaque prédiction = transaction PostgreSQL immédiate
- **Dégradation gracieuse** : Si PostgreSQL indisponible, prédictions continuent (logs erreur uniquement)

## 3. Composants techniques

### MLModel (app/models/ml_model.py)

**Chargement des artefacts ML (au démarrage de l'API) :**
```python
# Une seule fois au démarrage (lifespan FastAPI)
self.model = joblib.load('app/models/trained_model.pkl')  # XGBoost
self.onehot_encoder = joblib.load('app/models/onehot_encoder.pkl')
self.ordinal_encoder = joblib.load('app/models/ordinal_encoder.pkl') 
self.final_column_names = joblib.load('app/models/final_column_names.pkl')
```

**Preprocessing adapté au Projet 4 :**
- Encodage reproductible avec encodeurs pré-entraînés
- Gestion des valeurs Pydantic → format dataset d'origine
- Réorganisation automatique selon `final_column_names` (40 features)
- Validation cohérence features avant prédiction

**Prédiction optimisée :**
- Seuil décision : **0.514** (optimisé pour F1-Score maximal)
- Probabilités brutes XGBoost → décision binaire
- Facteurs risque heuristiques (améliorable avec SHAP)

### Base de données PostgreSQL (mode local)

**Architecture relationnelle (6 tables principales) :**

```sql
employees (1470 enregistrements dataset Projet 4)
├── prediction_requests (FK employee_id) 
    ├── prediction_results (1:1 avec request)
    └── prediction_sessions (groupe logique)
        └── api_audit_logs (traçabilité complète)
model_metadata (versioning ML)
```

**Connexions et performance :**
- **Gestionnaire** : `DatabaseManager` avec pool de connexions SQLAlchemy
- **Mode local** : PostgreSQL sur localhost (pas de cloud pour ce projet)
- **Transactions** : Auto-commit par défaut, rollback automatique si erreur
- **Index optimisés** : Sur `created_at`, `prediction`, `session_id` pour requêtes rapides

### Middleware de traçabilité

**PredictionLoggerMiddleware (app/middleware/prediction_logger.py) :**
- **Interception** : Tous les endpoints `/api/v1/predict/*`
- **Capture automatique** : Headers HTTP, payload, réponse, temps traitement
- **Session management** : UUID unique par utilisateur/batch
- **Mode dégradé** : Continue sans PostgreSQL (logs erreur uniquement)

**Données tracées automatiquement :**
```python
# Pour chaque prédiction
request_data = {
    "session_id": UUID,
    "input_data": employee_json,  # 27 variables complètes
    "client_ip": "client_real_ip", 
    "timestamp": "2025-09-11T..."
}

result_data = {
    "prediction": "Oui/Non",
    "probability_quit": 0.8547,
    "risk_factors": ["satisfaction faible", ...],
    "model_version": "1.0.0",
    "processing_time_ms": 45.2
}
```

## 4. Intégrations externes

### Hugging Face Spaces

**Configuration spécialisée :**
- **Point d'entrée** : `hf_app.py` (port 7860 requis HF)
- **Docker SDK** : Build automatique via Dockerfile
- **Variables d'environnement** : Gérées par settings.py
- **Limitations** : Pas de PostgreSQL cloud (base locale uniquement)

**Adaptation déploiement :**
```python
# hf_app.py - Wrapper pour HF Spaces
import uvicorn
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # HF Spaces requirement
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### CI/CD GitHub Actions

**Pipeline automatisé (deploy-hf-spaces.yml) :**
1. **Tests préalables** : Import FastAPI + dépendances
2. **Build conditionnel** : 
   - `develop` → environnement développement
   - `main` → production
3. **Déploiement** : Git push direct vers HF Space
4. **Vérification** : Health check post-déploiement

**Gestion des artefacts ML :**
- **Git LFS** : Fichiers .pkl volumineux (modèles, encodeurs)
- **Copie sélective** : Dossier `app/` complet vers HF Space
- **Nettoyage** : Suppression contenu existant avant redéploiement

## 5. Points d'attention

### Gestion des erreurs et mode dégradé

**Indisponibilité PostgreSQL :**
```python
# Mode dégradé gracieux
try:
    save_to_database(prediction_data)
except DatabaseError:
    logger.error("PostgreSQL indisponible - prédictions continuent")
    # API reste fonctionnelle sans traçabilité
```

**Échec chargement modèle :**
```python
# Au démarrage FastAPI
if not ml_model.is_loaded:
    # Erreur 503 pour tous les endpoints ML
    raise HTTPException(503, "Modèle XGBoost non disponible")
```

**Timeout et retry :**
- **Timeout prédiction** : 30 secondes (configurable)
- **Retry PostgreSQL** : 3 tentatives avec backoff exponentiel
- **Health checks** : `/health/` répond même si composants défaillants

### Performance et optimisations

**Temps de réponse mesurés :**
- **Prédiction individuelle** : 45ms moyenne (95e percentile: 80ms)
- **Batch 50 employés** : 1.2s (parallélisation possible)
- **Health check** : 5ms (réponse immédiate)

**Optimisations appliquées :**
- **Modèle pré-chargé** : Une seule fois au démarrage
- **Pool connexions** : SQLAlchemy optimisé
- **Validation Pydantic** : Cache automatique des schémas
- **Index PostgreSQL** : Requêtes sub-100ms

---

**Document généré** : septembre 2025  
**Version architecture** : 1.0.0 (Projet étudiant)  
**Stack validée** : FastAPI + XGBoost + PostgreSQL + Hugging Face Spaces  
**Statut déploiement** : Production opérationnelle