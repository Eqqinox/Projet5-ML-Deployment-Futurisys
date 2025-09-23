# Guide de Maintenance du Modèle ML - Futurisys

## Vue d'ensemble

Ce guide présente les procédures de base pour maintenir et surveiller le modèle XGBoost de prédiction d'attrition déployé. 
Architecture simple : FastAPI + PostgreSQL + Hugging Face Spaces.

## Monitoring de Base

### 1. Vérifications de Santé

#### Endpoints de Monitoring
```bash
# Vérification API
curl https://eqqinox-futurisys-ml-api.hf.space/health/

# Test modèle ML
curl -X POST https://eqqinox-futurisys-ml-api.hf.space/api/v1/predict/single \
  -H "Content-Type: application/json" \
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
}'
```

#### Métriques Simples à Surveiller
- API répond (status 200)
- Modèle chargé correctement
- Base PostgreSQL accessible
- Temps de réponse raisonnable (< 5 secondes)

### 2. Surveillance des Prédictions

#### Requête PostgreSQL Simple
```sql
-- Vérifier les prédictions récentes
SELECT 
    DATE(created_at) as jour,
    prediction,
    COUNT(*) as nombre
FROM prediction_results 
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(created_at), prediction
ORDER BY jour DESC;
```

#### Alerte Basique
```python
# Script simple de vérification
def check_prediction_distribution():
    # Récupérer prédictions derniers 7 jours
    quit_rate = get_quit_predictions_percentage(days=7)
    
    # Alerte si très différent du baseline (16%)
    if quit_rate < 0.05 or quit_rate > 0.30:
        print(f"ATTENTION: Taux démission anormal: {quit_rate:.1%}")
        # Envoyer email ou notification simple
```

## Mise à Jour du Modèle

### 1. Quand Réentraîner

#### Déclencheurs Simples
- **Trimestriel** : Mise à jour régulière avec nouvelles données
- **Performance dégradée** : Si précision chute visiblement
- **Nouvelles données** : Si dataset significativement enrichi

#### Vérification Performance
```python
# Test simple sur données de validation
def check_model_still_good():
    # Charger données test conservées
    test_data = load_test_dataset()
    
    # Prédictions actuelles
    predictions = model.predict(test_data)
    
    # Calculer accuracy basique
    accuracy = accuracy_score(test_data['target'], predictions)
    
    # Alerte si dégradation > 5%
    baseline_accuracy = 0.854  # Performance XGBoost
    if accuracy < baseline_accuracy - 0.05:
        print(f"ATTENTION: Performance dégradée: {accuracy:.3f}")
```

### 2. Processus de Réentraînement

#### Étape 1 : Préparation
```bash
# 1. Exporter nouvelles données depuis PostgreSQL
python database/export_new_data.py --since 2024-01-01

# 2. Combiner avec données historiques
python scripts/merge_datasets.py --old data/historical.csv --new data/new_data.csv

# 3. Validation rapide
python scripts/check_data_quality.py --file data/combined_dataset.csv
```

#### Étape 2 : Réentraînement
```python
# Script simple de réentraînement
def retrain_model():
    # 1. Charger données
    X, y = load_combined_training_data()
    
    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Utiliser même preprocessing que Projet 4
    preprocessor = load_existing_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 4. Réentraîner avec mêmes hyperparamètres
    model = XGBClassifier(
        n_estimators=200,
        scale_pos_weight=5.2,
        random_state=42,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.6,
        eval_metric='logloss',
        reg_alpha=0.1,
        reg_lambda=2
    )
    model.fit(X_train_processed, y_train)
    
    # 5. Valider performance
    accuracy = model.score(X_test_processed, y_test)
    if accuracy >= 0.80:  # Seuil minimum acceptable
        save_new_model_version(model, preprocessor)
        return True
    else:
        print(f"Performance insuffisante: {accuracy:.3f}")
        return False
```

### 3. Déploiement Simple

#### Via Git Push (Hugging Face Spaces)
```bash
# 1. Remplacer fichiers modèle
cp models/new_trained_model.pkl app/models/trained_model.pkl
cp models/new_preprocessor.pkl app/models/onehot_encoder.pkl

# 2. Tester localement
python -m uvicorn app.main:app --reload

# 3. Commit et push
git add app/models/
git commit -m "Update model v1.1 - retrained with new data"
git push origin main

# 4. Vérifier déploiement
curl https://eqqinox-futurisys-ml-api.hf.space/health/model
```

## Procédures de Rollback

### 1. Rollback Git Simple

#### Si Problème Après Déploiement
```bash
# 1. Identifier dernier commit stable
git log --oneline -n 5

# 2. Rollback vers version précédente
git revert HEAD

# 3. Push rollback
git push origin main

# 4. Vérifier que ça marche
curl https://eqqinox-futurisys-ml-api.hf.space/health/
```

### 2. Validation Post-Rollback
```python
# Tests basiques après rollback
def test_api_after_rollback():
    tests = {
        'api_health': test_health_endpoint(),
        'model_loads': test_model_loading(),
        'prediction_works': test_single_prediction(),
        'database_connects': test_db_connection().  #Base en local dans mon cas 
    }
    
    failed = [name for name, result in tests.items() if not result]
    
    if failed:
        print(f"Tests échoués: {failed}")
    else:
        print("Rollback réussi - système opérationnel")
```

## Maintenance Basique

### 1. Nettoyage PostgreSQL

#### Script Mensuel
```sql
-- Archiver anciennes prédictions (> 6 mois)
CREATE TABLE IF NOT EXISTS prediction_results_archive AS 
SELECT * FROM prediction_results WHERE created_at < CURRENT_DATE - INTERVAL '6 months';

-- Supprimer données archivées
DELETE FROM prediction_results 
WHERE created_at < CURRENT_DATE - INTERVAL '6 months';

-- Optimiser table
VACUUM ANALYZE prediction_results;
```

### 2. Sauvegarde

#### Sauvegarde Hebdomadaire
```bash
#!/bin/bash
# Script backup_weekly.sh

DATE=$(date +%Y%m%d)

# 1. Backup base de données
pg_dump futurisys_ml > backups/db_backup_$DATE.sql

# 2. Backup modèles (via Git)
git tag "backup-$DATE"
git push origin --tags

# 3. Nettoyer anciens backups (garde 4 semaines)
find backups/ -name "*.sql" -mtime +28 -delete

echo "Backup terminé: $DATE"
```

## Monitoring basique

### 1. Checklist Hebdomadaire
- [ ] API répond correctement
- [ ] Modèle fait des prédictions cohérentes  
- [ ] Base de données accessible - si pas en local
- [ ] Espace disque suffisant
- [ ] Tests automatisés passent

### 2. Script de Vérification Globale
```python
# weekly_check.py - Script simple de vérification
def weekly_system_check():
    print("=== VERIFICATION HEBDOMADAIRE ===")
    
    # 1. Test API
    api_ok = test_api_health()
    print(f"API Status: {'OK' if api_ok else 'ERREUR'}")
    
    # 2. Test modèle
    model_ok = test_model_prediction()
    print(f"Modèle ML: {'OK' if model_ok else 'ERREUR'}")
    
    # 3. Test base
    db_ok = test_database_connection()
    print(f"PostgreSQL: {'OK' if db_ok else 'ERREUR'}")
    
    # 4. Tests automatisés
    tests_ok = run_basic_tests()
    print(f"Tests: {'OK' if tests_ok else 'ERREUR'}")
    
    # Résumé
    all_ok = all([api_ok, model_ok, db_ok, tests_ok])
    print(f"\nStatut Global: {'TOUT FONCTIONNE' if all_ok else 'PROBLEMES DETECTES'}")

if __name__ == "__main__":
    weekly_system_check()
```

**Version guide** : 1.0 - Projet étudiant  
**Dernière mise à jour** : Septembre 2025  
**Contexte** : Formation OpenClassrooms Data Scientist et Machine Learning