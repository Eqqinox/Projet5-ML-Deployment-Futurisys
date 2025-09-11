---
title: Futurisys ML API
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: hf_app.py
pinned: false
---

# Futurisys ML API

**Déploiement d'un modèle XGBoost de prédiction d'attrition avec FastAPI, PostgreSQL et CI/CD**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.4+-orange.svg)](https://xgboost.readthedocs.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://postgresql.org)
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-blue.svg)](https://github.com/features/actions)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue.svg)](https://huggingface.co/spaces/Eqqinox/futurisys-ml-api)

## **Table des matières**

- [ À propos du projet](#-à-propos-du-projet)
- [ Architecture](#️-architecture)
- [ Démarrage rapide](#-démarrage-rapide)
- [ Installation détaillée](#️-installation-détaillée)
- [ Utilisation de l'API](#-utilisation-de-lapi)
- [ Tests et qualité](#-tests-et-qualité)
- [ Déploiement](#-déploiement)
- [ Configuration avancée](#-configuration-avancée)
- [ Contribution](#-contribution)
- [ Licence](#-licence)

## **À propos du projet**

### Contexte métier
L'**API Futurisys ML** est une solution de classification automatique développée pour identifier les causes d'attrition au sein d'une ESN (Entreprise de Services Numériques). Elle utilise un modèle **XGBoost** pré-entraîné pour prédire la probabilité qu'un employé quitte l'entreprise.

### Problème résolu
- **Prédiction d'attrition** : Identifier les employés à risque de départ
- **Explicabilité** : Comprendre les facteurs influençant les décisions
- **Traçabilité** : Historique complet des prédictions pour audit
- **Intégration** : API REST facilement intégrable dans les systèmes RH

### Fonctionnalités principales

✅ **API REST complète** avec documentation automatique (Swagger/OpenAPI)  
✅ **Modèle ML optimisé** : XGBoost avec seuil ajusté (0.514) pour F1-Score optimal  
✅ **Base de données** : PostgreSQL avec traçabilité complète des prédictions  
✅ **Tests automatisés** : Couverture 52% avec Pytest  
✅ **CI/CD intégré** : Pipeline GitHub Actions → Hugging Face Spaces  
✅ **Monitoring** : Endpoints de santé et métriques de performance  
✅ **Explicabilité** : Facteurs de risque identifiés pour chaque prédiction  

### Performances du modèle

| Métrique | Valeur | Description |
|----------|--------|-------------|
| **Accuracy** | 85.88% | Précision globale sur données de test |
| **F1-Score** | 56.56% | Équilibre précision/rappel |
| **Precision** | 56.54% | Taux de vrais positifs parmi les prédictions positives |
| **Recall** | 56.84% | Taux de détection des vrais cas d'attrition |
| **ROC-AUC** | 82.52% | Capacité de discrimination entre les classes |

## **Architecture**

### Vue d'ensemble
**Architecture en couches :**

**<u>Couche Présentation :</u>**
- Client HTTP → FastAPI Application

**<u>Couche Logique Métier :</u>**
- FastAPI → XGBoost Model
- FastAPI → Middleware Logger  
- FastAPI → PostgreSQL Database

**<u>Couche Data Processing :</u>**
- XGBoost Model → Preprocessing Pipeline
  - OneHot Encoder (variables catégorielles)
  - Ordinal Encoder (fréquence déplacement)

**<u>Couche Persistance :</u>**
- PostgreSQL → Prediction Sessions
- PostgreSQL → Prediction History
- Middleware Logger → Audit Trail

### Stack technique

#### **Backend & API**
- **[FastAPI 0.104+](https://fastapi.tiangolo.com)** - Framework web async/await haute performance
- **[Pydantic 2.5+](https://pydantic.dev)** - Validation automatique des données avec types Python
- **[Uvicorn](https://www.uvicorn.org)** - Serveur ASGI pour applications asynchrones

#### **Base de données & ORM**
- **[PostgreSQL 13+](https://postgresql.org)** - Base de données relationnelle ACID
- **[SQLAlchemy 2.0+](https://sqlalchemy.org)** - ORM Python (Object-Relational Mapping)
- **[Alembic 1.13+](https://alembic.sqlalchemy.org)** - Migrations de schéma versionnées

#### **Machine Learning & Data Science**
- **[XGBoost 3.0.4+](https://xgboost.readthedocs.io)** - Gradient Boosting optimisé (modèle principal)
- **[Scikit-learn 1.7.1+](https://scikit-learn.org)** - Preprocessing, métriques et pipeline ML
- **[Pandas 2.3.1+](https://pandas.pydata.org)** - Manipulation et analyse de données
- **[NumPy 1.26.0](https://numpy.org)** - Calculs numériques (version fixée pour compatibilité)

#### **Tests & Qualité de code**
- **[Pytest 7.4+](https://pytest.org)** - Framework de tests moderne
- **[Pytest-cov 4.1+](https://pytest-cov.readthedocs.io)** - Rapport de couverture de code

#### **DevOps & Déploiement**
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD avec tests automatisés
- **[Docker](https://docker.com)** - Conteneurisation pour déploiement reproductible
- **[Hugging Face Spaces](https://huggingface.co/spaces)** - Plateforme de déploiement ML

### Justifications techniques

#### Pourquoi FastAPI ?
- **Performance** : 300% plus rapide que Flask grâce à l'async/await natif
- **Documentation automatique** : Swagger/OpenAPI généré automatiquement
- **Validation stricte** : Intégration native avec Pydantic pour la validation de données
- **Écosystème moderne** : Support natif des type hints Python 3.6+

#### Pourquoi XGBoost ?
- **Performance ML excellente** : Algorithme gagnant de nombreuses compétitions Kaggle
- **Gestion native des valeurs manquantes** : Robustesse sur données réelles
- **Explicabilité intégrée** : Feature importance native
- **Scalabilité** : Optimisé pour les datasets de taille entreprise

#### Pourquoi PostgreSQL ?
- **ACID compliance** : Garanties transactionnelles pour audit trail
- **Requêtes complexes** : Support SQL avancé pour analytics
- **JSON natif** : Stockage flexible des métadonnées de prédictions
- **Performance** : Index B-tree et GIN pour requêtes ML

## **Démarrage rapide**

### Tester l'API en ligne

L'API est déployée sur Hugging Face Spaces et accessible immédiatement :

**Interface web interactive :** https://huggingface.co/spaces/Eqqinox/futurisys-ml-api

**Informations API :** https://eqqinox-futurisys-ml-api.hf.space/info

### Essai rapide avec curl

```bash
# Test de santé de l'API
curl -X 'GET' \
'https://eqqinox-futurisys-ml-api.hf.space/health/' \
  -H 'accept: application/json'

# Prédiction d'exemple
curl -X 'POST' \
  'https://eqqinox-futurisys-ml-api.hf.space/api/v1/predict/single' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "satisfaction_employee_environnement": 4,
  "satisfaction_employee_nature_travail": 4,
  "satisfaction_employee_equipe": 3,
  "satisfaction_employee_equilibre_pro_perso": 4,
  "note_evaluation_precedente": 4,
  "note_evaluation_actuelle": 4,
  "niveau_hierarchique_poste": 3,
  "heure_supplementaires": "Non",
  "augementation_salaire_precedente": 0.18,
  "age": 35,
  "genre": "Homme",
  "revenu_mensuel": 4500,
  "statut_marital": "Marié(e)",
  "departement": "Consulting",
  "poste": "Senior Manager",
  "nombre_experiences_precedentes": 3,
  "annee_experience_totale": 12,
  "annees_dans_l_entreprise": 5,
  "annees_dans_le_poste_actuel": 3,
  "annees_depuis_la_derniere_promotion": 2,
  "annes_sous_responsable_actuel": 3,
  "nombre_participation_pee": 2,
  "nb_formations_suivies": 4,
  "distance_domicile_travail": 8,
  "niveau_education": 5,
  "domaine_etude": "Marketing",
  "frequence_deplacement": "Voyage_Rare"
}
'
```

## **Installation détaillée**

### Prérequis système

- **Python 3.12** (requis pour compatibilité XGBoost/NumPy)
- **PostgreSQL 13+** (base de données principale)
- **Git** (pour clonage et versioning)
- **Docker** (Déploiement containerisé)

### Installation locale

#### 1. Clonage du repository

```bash
git clone https://github.com/Eqqinox/Projet5-ML-Deployment-Futurisys.git
cd Projet5
```

#### 2. Configuration de l'environnement Python

```bash
# Création de l'environnement virtuel
python3 -m venv venv

# Activation (Linux/macOS)
source venv/bin/activate

# Activation (Windows)
venv\Scripts\activate

# Installation des dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Configuration PostgreSQL

```bash
# -----------------------------------------------------------
# Ubuntu / Debian
# -----------------------------------------------------------
sudo apt update && sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createdb futurisys_ml
sudo -u postgres createuser --pwprompt futurisys_user

# -----------------------------------------------------------
# macOS (via Homebrew)
# -----------------------------------------------------------
brew install postgresql
brew services start postgresql
createdb futurisys_ml
createuser --pwprompt futurisys_user

# -----------------------------------------------------------
# Windows
# -----------------------------------------------------------
# 1. Télécharger et lancer l’installateur EDB : 
https://www.postgresql.org/download/windows/
# 2. Ajouter « C:\Program Files\PostgreSQL\<ver>\bin » au PATH
# 3. Dans psql ou pgAdmin :
CREATE DATABASE futurisys_ml;
CREATE USER futurisys_user WITH PASSWORD 'ton_mdp';
GRANT ALL PRIVILEGES ON DATABASE futurisys_ml TO futurisys_user;
```

#### 4. Variables d'environnement

```bash
# Copier le template à la racine
cp .env.example .env

# Éditer les variables
nano .env
```

**Contenu minimal du fichier `.env` :**
```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/futurisys_ml
POSTGRES_USER=futurisys_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=futurisys_ml
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
SECRET_KEY=your-secret-key-change-this-in-production

# ML Model Configuration
MODEL_PATH=app/models/trained_model.pkl
MODEL_VERSION=1.0.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Environment
ENVIRONMENT=development
```

#### 5. Initialisation de la base de données

```bash
# Méthode automatisée
cd database
python create_db.py

# Vérification de la création
python create_db.py --info
```

#### 6. Import des données (optionnel)

```bash
# Import du dataset de test
cd database
python import_dataset.py --file ../data/raw/dataset_projet4.csv
```

#### 7. Démarrage de l'application

```bash
# Retour à la racine du projet
cd ..

# Lancement du serveur de développement
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**L'API est maintenant accessible sur :** http://localhost:8000 (Redirection vers la documentation)

**Documentation interactive :** http://localhost:8000/docs

## **Utilisation de l'API**

### Endpoints principaux

#### **Santé de l'API**
- `GET /health/` - Statut de l'API

#### **Prédictions Machine Learning**
- `POST /api/v1/predict/single` - Prédiction individuelle d'employé
- `POST /api/v1/predict/batch` - Prédictions multiples (maximum 100 employés)
- `POST /api/v1/predict/validate-input` - Validation des données sans prédiction
- `GET /api/v1/predict/supported-values` - Valeurs acceptées pour chaque champ

#### 📊 **Informations API**
- `GET /info` - – Informations générales

#### **Données**
- `GET /api/v1/data/employees/count` - Nombre d'employés en base de données
- `GET /api/v1/data/predictions/history` - Historique des prédictions

#### **Analytiques**
- `GET /api/v1/analytics/predictions/stats` - Statistiques des prédictions

### Exemples d'utilisation

#### Traitement pour un candidat (single)

```python
import requests
import json

# Configuration
API_BASE_URL = "http://localhost:8000"  # ou l'URL Hugging Face

# Données d'exemple 1 - Profil stable (faible risque)
{
  "satisfaction_employee_environnement": 4,
  "satisfaction_employee_nature_travail": 4,
  "satisfaction_employee_equipe": 3,
  "satisfaction_employee_equilibre_pro_perso": 4,
  "note_evaluation_precedente": 4,
  "note_evaluation_actuelle": 4,
  "niveau_hierarchique_poste": 3,
  "heure_supplementaires": "Non",
  "augementation_salaire_precedente": 0.18,
  "age": 35,
  "genre": "Homme",
  "revenu_mensuel": 4500,
  "statut_marital": "Marié(e)",
  "departement": "Consulting",
  "poste": "Senior Manager",
  "nombre_experiences_precedentes": 3,
  "annee_experience_totale": 12,
  "annees_dans_l_entreprise": 5,
  "annees_dans_le_poste_actuel": 3,
  "annees_depuis_la_derniere_promotion": 2,
  "annes_sous_responsable_actuel": 3,
  "nombre_participation_pee": 2,
  "nb_formations_suivies": 4,
  "distance_domicile_travail": 8,
  "niveau_education": 5,
  "domaine_etude": "Marketing",
  "frequence_deplacement": "Voyage_Rare"
}

# Données d'exemple 2 - Profil à risque (fort risque)
{
  "satisfaction_employee_environnement": 1,
  "satisfaction_employee_nature_travail": 1,
  "satisfaction_employee_equipe": 2,
  "satisfaction_employee_equilibre_pro_perso": 1,
  "note_evaluation_precedente": 1,
  "note_evaluation_actuelle": 2,
  "niveau_hierarchique_poste": 1,
  "heure_supplementaires": "Oui",
  "augementation_salaire_precedente": 0.11,
  "age": 22,
  "genre": "Femme",
  "revenu_mensuel": 1800,
  "statut_marital": "Célibataire",
  "departement": "Commercial",
  "poste": "Représentant Commercial",
  "nombre_experiences_precedentes": 0,
  "annee_experience_totale": 1,
  "annees_dans_l_entreprise": 1,
  "annees_dans_le_poste_actuel": 1,
  "annees_depuis_la_derniere_promotion": 1,
  "annes_sous_responsable_actuel": 1,
  "nombre_participation_pee": 0,
  "nb_formations_suivies": 0,
  "distance_domicile_travail": 25,
  "niveau_education": 2,
  "domaine_etude": "Autre",
  "frequence_deplacement": "Voyage_Fréquent"
}

# Appel de l'endpoint de prédiction individuelle
# Envoie les données de l'employé au modèle XGBoost pour analyse
response = requests.post(
    f"{API_BASE_URL}/api/v1/predict/single",
    json=employee_data,
    headers={"Content-Type": "application/json"}
)

# Analyse de la réponse HTTP
if response.status_code == 200:
    # Réponse réussie : désérialisation JSON et extraction des données métier
    result = response.json()
    print(f"Prédiction: {result['prediction']}")
    print(f"Probabilité de départ: {result['probability_quit']:.2%}")
    print(f"Niveau de confiance: {result['confidence_level']}")
    print("Facteurs de risque:", ", ".join(result['risk_factors']))
else:
    # Gestion des erreurs : codes HTTP 4xx/5xx avec message d'erreur
    print(f"Erreur: {response.status_code} - {response.text}")
```

#### Traitement par lots (batch)

```python
import requests

# Exemple de liste d'employés à analyser
{
  "employees": [
    {
      "satisfaction_employee_environnement": 4,
      "satisfaction_employee_nature_travail": 4,
      "satisfaction_employee_equipe": 3,
      # ... autres variable à ajouter, voir exemple single ci-dessus)
    },
    {
      "satisfaction_employee_environnement": 1,
      "satisfaction_employee_nature_travail": 1,
      "satisfaction_employee_equipe": 2,
      # ... autres variable à ajouter, voir exemple single ci-dessus)
    }
  ]
  # Maximum 100 employés par requête
}

response = requests.post(
    f"{API_BASE_URL}/api/v1/predict/batch",
    json={"employees": employees}
)

# Analyse de la réponse HTTP
if response.status_code == 200:
    # Réponse réussie : désérialisation JSON et extraction des données métier
    results = response.json()
    for i, prediction in enumerate(results["predictions"]):
        print(f"Employé {i+1}: {prediction['prediction']} "
              f"({prediction['probability_quit']:.1%} de risque)")
else:
    # Gestion des erreurs : codes HTTP 4xx/5xx avec message d'erreur
    print(f"Erreur: {response.status_code} - {response.text}")
```

### Format des réponses

#### Exemple de réponse de prédiction individuelle
```json
Response body
Download
{
  "employee_id": null,
  "prediction": "Non",
  "probability_quit": 0.005100000184029341,
  "probability_stay": 0.9948999881744385,
  "confidence_level": "Élevé",
  "risk_factors": [],
  "model_version": "1.0.0",
  "timestamp": "2025-09-11T10:35:10.614423"
}
```

#### Exemple de réponse de prédiction par lots (batch)
```json
{
  "predictions": [
    {
      "employee_id": null,
      "prediction": "Non",
      "probability_quit": 0.005100000184029341,
      "probability_stay": 0.9948999881744385,
      "confidence_level": "Élevé",
      "risk_factors": [],
      "model_version": "1.0.0",
      "timestamp": "2025-09-11T10:27:42.528620"
    },
    {
      "employee_id": null,
      "prediction": "Oui",
      "probability_quit": 0.9979000091552734,
      "probability_stay": 0.002099999925121665,
      "confidence_level": "Élevé",
      "risk_factors": [
        "Satisfaction environnement très faible",
        "Satisfaction travail très faible",
        "Heures supplémentaires fréquentes"
      ],
      "model_version": "1.0.0",
      "timestamp": "2025-09-11T10:27:42.536020"
    }
  ],
  "total_employees": 2,
  "quit_predictions": 1,
  "stay_predictions": 1,
  "average_quit_probability": 0.5015,
  "processing_time_seconds": 0.026
}
```

#### Exemple de réponse d'erreur de validation
```json
{
  "detail": [
    {
      "loc": [
        "string",
        0
      ],
      "msg": "string",
      "type": "string"
    }
  ]
}
```

## **Tests et qualité**

### Structure des tests

```
tests/
├── test_api_endpoints.py      # Tests unitaires (34 tests)
├── test_integration.py        # Tests d'intégration end-to-end (19 tests) 
├── test_performance.py        # Tests de charge (17 tests)
└── test_edge_cases.py         # Tests des cas limites (15 tests)
```

### Lancement des tests

```bash
# Test complet avec couverture
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing

# Tests rapides (unitaires uniquement)
pytest tests/test_api_endpoints.py -v                         # Lance toutes les classes et fonctions de test présentes
pytest tests/test_api_endpoints.py::TestDataEndpoints -v      # Exécute uniquement les méthodes de la classe (liés aux routes /api/v1/data/*)
pytest tests/test_api_endpoints.py::TestAnalyticsEndpoints -v # Exécute uniquement les méthodes de la classe (liés aux routes /api/v1/analytics/*).

# Tests de performance
pytest tests/test_performance.py -v -m performance

# Tests d'intégration (nécessite PostgreSQL)
pytest tests/test_integration.py -v -m integration
```

### Métriques

- **Tests automatisés :** 85 tests développés (84 réussis, 1 échec acceptable de performance)
- **Couverture de code :** 52% globale (>90% sur modules critiques testables)
- **Tests unitaires :** 34 tests de validation des endpoints API
- **Tests fonctionnels :** 51 tests (intégration, performance, cas limites)
- **Architecture :** 4 fichiers organisés par fonction (unitaires, intégration, performance, edge cases)

### Rapport de couverture

Après exécution des tests, le rapport HTML est disponible dans `htmlcov/index.html`.

## **Déploiement**

### Hugging Face Spaces (Multi-environnements)
Le déploiement est automatisé via GitHub Actions :
1. **Push sur `develop`** → Déploiement en environnement de développement
2. **Pull Request vers `main`** → Déploiement de staging pour tests
3. **Push sur `main`** → Déploiement automatique en production

**URL de production :** https://huggingface.co/spaces/Eqqinox/futurisys-ml-api

### Déploiement Docker

**Configuration Hugging Face Spaces :**
```bash
# Le projet utilise Docker comme SDK sur Hugging Face Spaces
# Port 7860 requis par HF Spaces (configuré automatiquement)
# Point d'entrée : hf_app.py

# Pour tester localement le build HF Spaces :
docker build -t futurisys-ml-api .
docker run -p 7860:7860 futurisys-ml-api
```

### Pipeline CI/CD
Notre pipeline GitHub Actions automatise :
1. **Tests** : Validation FastAPI et chargement des dépendances
2. **Deploy** : Déploiement automatique vers Hugging Face Spaces selon la branche
3. **Monitoring** : Endpoints de santé intégrés (`/health`, `/health/model`)

## **Configuration avancée**

### Variables d'environnement complètes

```bash
# === Database Configuration ===
DATABASE_URL=postgresql://futurisys_user:secure_password@localhost:5432/futurisys_ml
POSTGRES_USER=futurisys_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=futurisys_ml
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# === API Configuration ===
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
SECRET_KEY=clé-secrète

# === ML Model Configuration ===
MODEL_PATH=app/models/trained_model.pkl
MODEL_VERSION=1.0.0
MAX_BATCH_SIZE=100
PREDICTION_TIMEOUT=30

# === Logging ===
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# === Environment ===
ENVIRONMENT=development

# === Security (pour usage futur) ===
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALGORITHM=HS256
```

### Configuration avancée PostgreSQL
```sql
-- Optimisations pour ML workloads
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';

-- Index optimisés pour analytics
CREATE INDEX CONCURRENTLY idx_predictions_created_at_partial 
ON prediction_results (created_at) 
WHERE created_at >= NOW() - INTERVAL '30 days';

CREATE INDEX CONCURRENTLY idx_predictions_model_version 
ON prediction_results USING HASH (model_version);
```

### Scaling et déploiement
Le déploiement s'effectue automatiquement via Hugging Face Spaces :
- **Infrastructure gérée** : Pas de configuration serveur requise
- **Scaling automatique** : Géré par la plateforme Hugging Face
- **Déploiement zero-downtime** : Via Git push automatique
- **Monitoring intégré** : Logs et métriques Hugging Face Spaces

### Workflow de développement
1. **Cloner** le repository : `git clone https://github.com/Eqqinox/Projet5-ML-Deployment-Futurisys.git`
2. **Créer une branche** : `git checkout -b feature/nouvelle-fonctionnalite`
3. **Développer** en suivant les conventions de code
4. **Tester** : `pytest tests/ --cov=app`
5. **Commit** : `git commit -m 'feat: ajouter nouvelle fonctionnalité'`
6. **Push** : `git push origin feature/nouvelle-fonctionnalite`
7. **Pull Request** vers `develop` pour tests automatiques
8. **Merge vers main** après validation pour déploiement production

### Standards de code
- **Tests :** 85 tests automatisés avec Pytest (52% de couverture)
- **Couverture :** Focalisée sur les endpoints critiques et cas métier
- **Validation :** Tests de performance, charge et cas limites
- **Architecture :** Tests d'intégration end-to-end
- **CI/CD :** Validation automatique via GitHub Actions

### Contexte académique
Projet réalisé dans le cadre de la formation **"Expert en ingénierie et science des données"** (Niveau 7) chez OpenClassrooms.

### Exemple d'utilisation pour étudiants
```bash
# Setup environnement de développement
git clone https://github.com/Eqqinox/Projet5-ML-Deployment-Futurisys.git
cd Projet5
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Tests et validation
pytest tests/ --cov=app
python database/create_db.py
uvicorn app.main:app --reload
```

## **Support et contact**

- **Auteur :** MMeknaci (mounir.meknaci@gmail.com)
- **Projet :** Formation Data Scientist Machine Learning - OpenClassrooms
- **Repository :** [GitHub](https://github.com/Eqqinox/projet5-ml-deployment)
- **Démo live :** [Hugging Face Spaces](https://huggingface.co/spaces/Eqqinox/futurisys-ml-api)

### Remerciements

- **OpenClassrooms** pour le parcours Data Science complet et le contexte métier et les spécifications
- **Communauté open source** : FastAPI, XGBoost, PostgreSQL
- **Hugging Face** pour la plateforme de déploiement gratuite

---

## **Métriques du modèle ML**
**Performances validées (Projet 4) :**
- **Accuracy** : 85.88%
- **Precision** : 56.54%
- **Recall** : 56.84%
- **F1-Score** : 56.56%
- **Seuil optimal** : 0.514

**Tests de performance API :**
- **85 tests automatisés** avec 52% de couverture
- **Temps de réponse** : < 200ms (tests locaux)
- **Support concurrent** : 50 utilisateurs (tests de charge)

---

**## Environnements de déploiement**

**### Développement local**
- **URL** : http://localhost:8000
- **Base** : PostgreSQL locale (optionnelle)
- **Démarrage** : `uvicorn app.main:app --reload`

**### Production (Hugging Face Spaces)**
- **URL** : https://huggingface.co/spaces/Eqqinox/futurisys-ml-api
- **Déploiement** : Automatique via GitHub Actions
- **Branches** : `develop` (dev) et `main` (production)
- **Monitoring** : Endpoints `/health` intégrés

---
*Dernière mise à jour : Septembre 2025*
*Statut du projet : ✅ Déployé et fonctionnel*