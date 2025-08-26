---
title: Futurisys ML API
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: hf_app.py
pinned: false
---

# Projet5 - ML Deployment Futurisys

**Déploiement d'un modèle de Machine Learning XGBoost avec FastAPI, PostgreSQL et CI/CD**

*Projet 5 du parcours Data Science Machine Learning - Déployer un modèle de Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.4+-orange.svg)](https://xgboost.readthedocs.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://postgresql.org)

## Description du projet

API de classification automatique d'informations (Identifiez les causes d'attrition au sein d'une ESN) développée pour **Futurisys** utilisant un modèle **XGBoost** pré-entraîné.

Ce projet implémente les meilleures pratiques de **Machine Learning Operations (MLOps)** avec :
- **API FastAPI** avec validation Pydantic
- **Modèle XGBoost** de classification (issu du Projet4)
- **Base de données PostgreSQL** pour la traçabilité des prédictions
- **Tests automatisés** avec Pytest et couverture de code
- **Pipeline CI/CD** avec GitHub Actions
- **Déploiement** sur Hugging Face Spaces
- **Explicabilité** des prédictions avec SHAP

## Technologies utilisées

### Backend & API
- **FastAPI 0.104+** - Framework web moderne et rapide
- **Pydantic 2.5+** - Validation des données
- **Uvicorn** - Serveur ASGI

### Base de données
- **PostgreSQL 13+** - Base de données relationnelle
- **SQLAlchemy 2.0+** - ORM Python
- **Alembic 1.13+** - Migrations de base de données

### Machine Learning
- **XGBoost 3.0.4+** - Modèle de classification gradient boosting
- **Scikit-learn 1.7.1+** - Préprocessing et métriques ML
- **Pandas 2.3.1+** - Manipulation des données
- **NumPy 1.26.0** - Calculs numériques (version fixe pour compatibilité)
- **SHAP 0.48+** - Explicabilité des prédictions
- **Imbalanced-learn 0.14+** - Gestion des classes déséquilibrées
- **Joblib 1.4+** - Sérialisation des modèles

### Tests & Qualité
- **Pytest 7.4+** - Framework de tests
- **Pytest-cov 4.1+** - Couverture de code
- **Black 23.11+** - Formatage du code
- **Flake8 6.1+** - Analyse statique

### DevOps
- **GitHub Actions** - CI/CD
- **Docker** - Conteneurisation
- **Hugging Face Spaces** - Déploiement

## Installation

### Prérequis
- **Python 3.12** (requis pour compatibilité avec le modèle XGBoost)
- **PostgreSQL 13+**
- **Git**

### Configuration locale

1. **Cloner le repository**
```bash
git clone https://github.com/Eqqinox/projet5-ml-deployment.git
cd Projet5# Test final infrastructure CI/CD
