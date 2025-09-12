# Documentation Technique du Modèle ML - XGBoost Attrition

## Vue d'ensemble

Ce document détaille le modèle de Machine Learning déployé pour la prédiction d'attrition des employés chez Futurisys. Le modèle utilise XGBoost (Extreme Gradient Boosting) pour classifier les employés selon leur probabilité de démission.

## Architecture du Modèle

### Type et Algorithme
- **Algorithme :** XGBoost (Extreme Gradient Boosting)
- **Type :** Classification binaire supervisée
- **Framework :** Python avec scikit-learn et XGBoost 3.0.4+
- **Variable cible :** Attrition (0 = Reste, 1 = Démission)

### Pipeline de Préprocessing

Le modèle utilise un pipeline de préprocessing complet comprenant :

1. **Encodage des variables catégorielles**
   - OneHot Encoder pour variables nominales (département, domaine_etude, etc.)
   - Ordinal Encoder pour variables ordinales (satisfaction_employee_*, etc.)

2. **Gestion des données numériques**
   - Pas de normalisation appliquée (XGBoost robuste aux échelles)
   - 40 features finales après encodage

3. **Gestion du déséquilibre**
   - Scale_pos_weight = 5.2 (ratio 84/16 des classes)
   - Validation croisée stratifiée pour préserver les proportions de classes
   - Métrique d'évaluation : logloss pour optimiser les probabilités

## Hyperparamètres Optimisés

```python
XGBClassifier(
    n_estimators=200,           # Nombre d'arbres
    scale_pos_weight=5.2,       # Gestion déséquilibre (84/16 ratio)
    random_state=42,            # Reproductibilité
    max_depth=3,                # Profondeur maximale des arbres
    learning_rate=0.1,          # Taux d'apprentissage
    subsample=0.6,              # Sous-échantillonnage lignes
    colsample_bytree=1,         # Sous-échantillonnage colonnes
    eval_metric='logloss',      # Optimisation probabilités
    reg_alpha=0.1,              # Régularisation L1 (Lasso)
    reg_lambda=2                # Régularisation L2 (Ridge)
)
```

### Justification des Hyperparamètres

- **scale_pos_weight=5.2** : Compense le déséquilibre 84% reste / 16% démission
- **max_depth=3** : Évite l'overfitting avec des arbres peu profonds
- **subsample=0.6** : Réduit l'overfitting par échantillonnage aléatoire
- **reg_alpha=0.1, reg_lambda=2** : Régularisation pour généralisation

## Performances Détaillées

### Métriques de Validation Croisée stratifiée (5-folds)
```
Résultats de validation croisée stratifiée (5 folds):
Accuracy  : 0.8588 (± 0.0220)
Precision : 0.5654 (± 0.0701)
Recall    : 0.5684 (± 0.0678)
F1-Score  : 0.5656 (± 0.0638)
ROC-AUC   : 0.8252 (± 0.0212)
```

### Performances sur Test Set (seuil par défaut 0.5)
```
Accuracy (Exactitude):    0.8537
Precision (Précision):    0.5400
Recall (Rappel):          0.5745
F1-Score:                 0.5567
ROC-AUC:                  0.8270
```

### Matrice de Confusion (Test Set)
```
                Prédictions
Réalité    Reste  Démission
Reste       224       23
Démission    20       27
```

## Optimisation du Seuil de Décision

### Analyse des Seuils
Le modèle utilise un seuil optimisé basé sur l'analyse de la courbe Precision-Recall :

```
Seuil optimal (F1-Score max): 0.5136
F1-Score correspondant: 0.5745
Precision correspondante: 0.5745
Recall correspondant: 0.5745
```

### Justification du Seuil Optimal

1. **Seuil retenu :** 0.5136
2. **Critère d'optimisation :** Maximisation du F1-Score
3. **Amélioration :** +0.0178 de gain F1-Score vs seuil par défaut
4. **ROC-AUC élevée (0.827)** confirme la validité de l'optimisation

### Performances avec Seuil Optimisé (0.5136)
```
Accuracy:  0.8639 (+0.0102)
Precision: 0.5745 (+0.0345)
Recall:    0.5745 (+0.0000)
F1-Score:  0.5745 (+0.0178)

Matrice de Confusion Optimisée:
[[227  20]
[ 20  27]]
```

## Features et Importance

### Top 10 Features les Plus Importantes
```
1. heure_supplementaires                  0.741383
2. nombre_participation_pee               0.467252
3. nombre_experiences_precedentes         0.446803
4. revenu_mensuel                         0.424747
5. distance_domicile_travail              0.404249
6. age                                    0.374470
7. satisfaction_employee_environnement    0.349192
8. satisfaction_employee_nature_travail   0.332217
9. departement_Consulting                 0.331896
10. annes_sous_responsable_actuel         0.328836
```

### Analyse SHAP - Explicabilité

Le modèle intègre une analyse SHAP pour l'explicabilité :

1. **Impact global** : Les heures supplémentaires sont le facteur le plus discriminant
2. **Interactions** : Les features interagissent de manière complexe
3. **Prédictions individuelles** : Chaque prédiction peut être expliquée feature par feature

## Structure des Données d'Entrée

### Format d'Input API
```json
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
```

### Contraintes de Validation
- **satisfaction_employee_*** : Entiers 1-4
- **age** : 18-60 ans
- **revenu_mensuel** : 1000-20000 €
- **distance_domicile_travail** : 1-29 km
- **Variables catégorielles** : Valeurs prédéfinies selon dataset Projet 4

## Format de Sortie API

### Réponse Standard
```json
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

### Niveaux de Confiance
- **Élevé** : Probabilité > 0.7
- **Moyen** : Probabilité 0.4-0.7  
- **Faible** : Probabilité < 0.4

## Métriques de Performance Production

### Temps de Réponse
- **Temps moyen** : 45ms
- **95e percentile** : 80ms
- **Timeout** : 5000ms

### Métriques de Qualité
- **Accuracy attendue** : 86.4% (± 2%)
- **F1-Score attendu** : 57.5% (± 5%)
- **ROC-AUC attendue** : 82.7% (± 2%)

## Limitations et Considérations

### Limitations Techniques
1. **Dataset déséquilibré** : 84% reste / 16% démission
2. **Généralisation** : Entraîné sur données ESN spécifiques
3. **Features temporelles** : Pas de features de séries temporelles
4. **Biais potentiels** : Données historiques peuvent contenir des biais

### Considérations Métier
1. **Interprétation probabiliste** : Utiliser les probabilités, pas seulement la classe
2. **Facteurs humains** : Le modèle ne capture pas tous les facteurs de démission
3. **Évolution temporelle** : Les patterns peuvent évoluer dans le temps
4. **Action préventive** : Utiliser pour identification précoce, pas décision finale

## Versioning et Traçabilité

### Version Actuelle
- **Version modèle** : xgboost_v1.0
- **Date d'entraînement** : Projet 4 (données historiques ESN)
- **Features** : 40 variables après preprocessing (OneHot + Ordinal)
- **Algorithme** : XGBoost 3.0.4 avec hyperparamètres optimisés
- **Dataset source** : 1470 employés ESN avec 27 variables d'origine

### Architecture de Traçabilité PostgreSQL

#### Tables Principales
```sql
-- Employés (référentiel métier)
employees: 1470 enregistrements du dataset Projet 4

-- Sessions de prédiction (regroupement logique)
prediction_sessions: Identifiant unique par utilisateur/batch

-- Requêtes de prédiction (inputs détaillés)
prediction_requests: Toutes les données d'entrée du modèle

-- Résultats de prédiction (outputs complets)
prediction_results: Prédictions + probabilités + facteurs de risque

-- Métadonnées des modèles (versioning ML)
model_metadata: Versions, performances, configurations
```

#### Données Tracées Automatiquement
- **Données d'entrée** : 27 variables employé complètes
- **Résultats ML** : Prédiction binaire + probabilités (quit/stay)
- **Métadonnées** : Version modèle, seuil utilisé (0.5136), temps traitement
- **Contexte système** : Timestamp UTC, IP utilisateur, session ID
- **Explicabilité** : Facteurs de risque identifiés par feature importance

#### Middleware de Traçabilité
```python
# Capture automatique transparente
PredictionLoggerMiddleware: 
  - Intercepte tous les appels /predict/*
  - Sauvegarde input/output en PostgreSQL
  - Mode dégradé si BDD indisponible
  - Logging asynchrone pour performance
```

## Tests et Validation

### Architecture de Tests (85 tests développés)

#### Tests Unitaires (34 tests)
```bash
tests/test_api_endpoints.py
- Validation endpoints FastAPI individuellement
- Schémas Pydantic : 100% couverture
- Gestion d'erreurs : codes HTTP appropriés
- Performance : < 200ms par endpoint
```

#### Tests d'Intégration (19 tests)
```bash
tests/test_integration.py
- Workflow complet API → ML → PostgreSQL
- Cohérence données input/output
- Stabilité modèle : reproductibilité garantie
- Spécifications OpenAPI validées
```

#### Tests de Performance (17 tests)
```bash
tests/test_performance.py
- Temps réponse : Health <100ms, Prédiction <200ms
- Charge simultanée : 50 utilisateurs supportés
- Stabilité : 10 secondes sous contrainte
- Timeout : 30 secondes maximum configuré
```

#### Tests Cas Limites (15 tests)
```bash
tests/test_edge_cases.py
- Données réelles Projet 4 : employés haut/faible risque
- Valeurs extrêmes et corrompues
- Validation robustesse métier
- Edge cases modèle ML
```

### Métriques de Qualité Validées

#### Couverture de Code (51% globale - 929 lignes analysées)
<u>Modules à couverture élevée (>90%) :</U>
- **Schémas Pydantic** : 100% (validation critique)
- **Modèles BDD** : 95% (intégrité données) 
- **Configuration** : 100% (paramétrage)
- **Health endpoints** : 100% (monitoring)

<u>Modules à couverture moyenne (50-70%) :</u>
- **Main application** : 51% (logique FastAPI)
- **Middleware traçabilité** : 54% (logging PostgreSQL)
- **Analytics endpoints** : 50% (rapports métier)

<u>Modules à couverture faible (<30%)</u> :
- **Modèle ML** : 15% (wrapper XGBoost complexe)
- **Connexions BDD** : 16% (gestion erreurs PostgreSQL)

<u>Note importante :</u> La faible couverture des modules ML et BDD reflète la complexité de tester les interactions avec des systèmes externes (PostgreSQL, modèles sérialisés). Les parties critiques pour la logique métier (validation, endpoints, configuration) atteignent 90-100% de couverture.

#### Performances Mesurées (85 tests exécutés en 12.47s)

- **Prédiction individuelle** : Temps variable selon charge système
- **Validation données** : Rapide (<50ms estimé)
- **Health checks** : Très rapide (<10ms estimé)
- **Suite complète** : 12.47s pour 85 tests (performance acceptable)

#############################
### Monitoring Production Opérationnel

#### Performances Mesurées (85 tests exécutés en 12.47s)
- **Prédiction individuelle** : 45ms moyenne (95e percentile: 80ms)
- **Prédiction batch (50 employés)** : 1.2s moyenne
- **Validation données** : 15ms moyenne
- **Health checks** : 5ms moyenne

### Monitoring Production Opérationnel

#### Surveillance Automatique PostgreSQL
```python
# Métriques base de données temps réel
/api/v1/Analytics/predictions/stats:
  - Nombre prédictions par jour/semaine
  - Répartition démission/reste
  - Performance requêtes SQL
  - Espace disque utilisé

/api/v1/data/employees/count:
  - Nombre de personnels par département
  - Nombre de personnels par attrition

/api/v1/data/predictions/history:
  - Historique des prédictions réalisées
```

#### Health Checks
```python
GET /health/           # Status API basique
```

### Validation Continue

#### Tests de Régression Automatisés

- **CI/CD intégré** : 85 tests exécutés à chaque déploiement
- **Validation pre-prod** : Tests sur données réelles anonymisées