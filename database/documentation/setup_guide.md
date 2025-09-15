# Guide d'installation et utilisation - Base de données PostgreSQL

## 🎯 Vue d'ensemble

Ce guide vous accompagne pour créer et configurer la base de données PostgreSQL pour le projet Futurisys ML avec traçabilité complète des prédictions.

## 📋 Prérequis

### 1. Installation PostgreSQL locale

#### macOS (Homebrew)
```bash
# Installation PostgreSQL
brew install postgresql@13

# Démarrage du service
brew services start postgresql@13

# Création d'un utilisateur (optionnel)
createuser -s postgres
```

#### Ubuntu/Debian
```bash
# Installation
sudo apt update
sudo apt install postgresql postgresql-contrib

# Démarrage du service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### Windows
- Télécharger PostgreSQL depuis https://www.postgresql.org/download/windows/
- Suivre l'installateur graphique
- Noter le mot de passe du superutilisateur `postgres`

### 2. Configuration initiale

```bash
# Se connecter en tant que postgres
sudo -u postgres psql

# Ou sur Windows/macOS
psql -U postgres

# Créer la base de données et l'utilisateur
CREATE DATABASE futurisys_ml;
CREATE USER futurisys_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE futurisys_ml TO futurisys_user;

# Quitter psql
\q
```

### 3. Test de connexion

```bash
# Test avec l'utilisateur créé
psql -h localhost -U futurisys_user -d futurisys_ml

# Si succès, vous verrez:
# futurisys_ml=>
```

## ⚙️ Configuration des variables d'environnement

### Fichier `.env`
```bash
# Base de données locale
DATABASE_URL=postgresql://futurisys_user:secure_password@localhost:5432/futurisys_ml
POSTGRES_USER=futurisys_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=futurisys_ml
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Configuration API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Configuration ML
MODEL_PATH=app/models/trained_model.pkl
MODEL_VERSION=1.0.0
```

## 🚀 Utilisation des scripts

### Méthode unique : Approche ORM SQLAlchemy

```bash
# Installation des dépendances Python
pip install -r requirements.txt

# Import du dataset (crée automatiquement la structure)
python database/python/import_dataset.py --file dataset_projet4.csv

# Ou création manuelle via Python
python -c "
from app.database.models import Base
from app.database.connection import DatabaseManager
db = DatabaseManager()
Base.metadata.create_all(bind=db.engine)
print('Tables créées avec succès!')
"
```

#### Sortie attendue du script Python
```
🚀 Début de la création de la base de données Futurisys ML
============================================================
✅ Connexion PostgreSQL réussie: PostgreSQL 13.x
✅ Extensions créées avec succès
🏗️ Création des tables...
✅ Tables créées avec succès
📋 Tables créées: api_audit_logs, employees, model_metadata, prediction_requests, prediction_results, prediction_sessions
✅ 47 contraintes créées
✅ Vues utilitaires créées
✅ Métadonnées du modèle XGBoost ajoutées
✅ Session de test créée: 550e8400-e29b-41d4-a716-446655440000
============================================================
✅ Base de données créée avec succès !
🔗 URL de connexion: postgresql://futurisys_user:***@localhost:5432/futurisys_ml
📊 Tables créées: 6
🎯 La base est prête pour l'import du dataset et l'utilisation par l'API
```

## 📥 Import du dataset Projet 4

### Préparation du fichier CSV

Assurez-vous que votre fichier CSV du Projet 4 contient bien les 28 colonnes :
- 27 features + 1 variable cible `a_quitte_l_entreprise`
- 1470 lignes d'employés
- Aucune valeur manquante
- Encodage UTF-8

### Exécution de l'import

```bash
# Import automatique (recherche le fichier CSV)
python database/import_dataset.py

# Import avec fichier spécifique
python database/import_dataset.py --file /path/to/dataset_projet4.csv

# Import avec suppression des données existantes
python database/import_dataset.py --file dataset.csv --clear --yes

# Options avancées
python database/import_dataset.py \
    --file dataset.csv \
    --clear \
    --batch-size 50 \
    --report import_report.json \
    --url "postgresql://user:pass@localhost:5432/db"
```

#### Sortie attendue de l'import
```
🚀 Début de l'import du dataset Projet 4
============================================================
📖 Chargement du fichier: dataset_projet4.csv
📊 Données chargées: 1470 lignes, 28 colonnes
🔍 Validation de la structure du dataset...
✅ Structure du dataset validée
🧹 Validation et nettoyage des données...
✅ Validation des données réussie
🔧 Préparation des données pour l'import...
✅ 1470 enregistrements préparés
📥 Import de 1470 enregistrements par batch de 100...
✅ Batch 1: 100 enregistrements importés
✅ Batch 2: 100 enregistrements importés
...
✅ Batch 15: 70 enregistrements importés
📊 Import terminé: 1470 réussites, 0 erreurs
🔍 Vérification de l'import...
📊 Total employés en base: 1470
📈 Statistiques par département:
  Commercial: 961 employés, 163 démissions (17.0%)
  Consulting: 446 employés, 68 démissions (15.2%)
  Ressources Humaines: 63 employés, 6 démissions (9.5%)
👥 Âge: min=18, max=60, moy=36.9
💰 Salaire: min=1009, max=19999, moy=6503€
✅ Vérification réussie - données cohérentes
============================================================
✅ Import terminé avec succès !
📊 1470/1470 enregistrements importés
⏱️ Durée: 3.45 secondes
📈 Taux de réussite: 100.0%
🎉 Script d'import terminé avec succès !
```

## 🔍 Vérification et exploration

### Requêtes de vérification SQL

```sql
-- Nombre total d'employés
SELECT COUNT(*) as total_employees FROM employees;

-- Répartition par département
SELECT departement, COUNT(*) as count, 
       COUNT(CASE WHEN a_quitte_l_entreprise = 'Oui' THEN 1 END) as quits
FROM employees 
GROUP BY departement;

-- Statistiques des salaires
SELECT MIN(revenu_mensuel), MAX(revenu_mensuel), AVG(revenu_mensuel)
FROM employees;

-- Vérification des contraintes
SELECT 
    COUNT(CASE WHEN age < 18 OR age > 60 THEN 1 END) as invalid_age,
    COUNT(CASE WHEN revenu_mensuel < 1000 OR revenu_mensuel > 20000 THEN 1 END) as invalid_salary
FROM employees;
```

### Script de vérification Python

```python
# Vérification rapide avec Python
from app.database.models import Employee
from app.database.connection import DatabaseManager

db = DatabaseManager()
session = db.get_session()

# Comptages
total = session.query(Employee).count()
print(f"Total employés: {total}")

# Par département
from sqlalchemy import func
dept_stats = session.query(
    Employee.departement,
    func.count(Employee.employee_id)
).group_by(Employee.departement).all()

for dept, count in dept_stats:
    print(f"{dept}: {count} employés")

session.close()
```

## 🛠️ Dépannage

### Problèmes courants

#### 1. Erreur de connexion
```
psql: error: connection to server on socket "/tmp/.s.PGSQL.5432" failed
```
**Solution :** Vérifier que PostgreSQL est démarré
```bash
# macOS
brew services restart postgresql@13

# Linux
sudo systemctl restart postgresql

# Windows - via Services ou
net start postgresql-x64-13
```

#### 2. Permission refusée
```
FATAL: role "futurisys_user" does not exist
```
**Solution :** Créer l'utilisateur
```sql
-- Se connecter en tant que postgres
sudo -u postgres psql
CREATE USER futurisys_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE futurisys_ml TO futurisys_user;
```

#### 3. Base de données existe déjà
```
ERROR: database "futurisys_ml" already exists
```
**Solution :** Supprimer manuellement la base
```sql
DROP DATABASE IF EXISTS futurisys_ml;
CREATE DATABASE futurisys_ml;
```

#### 4. Fichier CSV non trouvé
```
FileNotFoundError: Fichier CSV non trouvé
```
**Solution :** Spécifier le chemin complet
```bash
python import_dataset.py --file /chemin/complet/vers/dataset.csv
```

#### 5. Erreur de validation des données
```
ValueError: Dataset contient des données invalides
```
**Solution :** Vérifier la structure et les valeurs du CSV
- Vérifier les plages de valeurs (âge, salaire, etc.)
- Contrôler les valeurs catégorielles
- S'assurer qu'il n'y a pas de valeurs manquantes

### Logs de débogage

```bash
# Activer les logs détaillés pour PostgreSQL
export LOG_LEVEL=DEBUG
python database/python/import_dataset.py --file dataset.csv

# Logs SQL visibles dans SQLAlchemy
# Modifier app/database/connection.py : echo=True
```

## 📊 Structure finale de la base

Après l'exécution réussie, votre base contient :

### Tables principales
- `employees` (1470 lignes) - Dataset Projet 4
- `prediction_sessions` - Sessions de prédiction
- `prediction_requests` - Inputs du modèle
- `prediction_results` - Outputs du modèle  
- `model_metadata` - Métadonnées des modèles
- `api_audit_logs` - Audit des appels API

### Vues utilitaires
- `v_predictions_summary` - Résumé des prédictions
- `v_employee_stats_by_dept` - Statistiques par département

### Index de performance
- 47 contraintes CHECK pour l'intégrité
- 15+ index pour optimiser les requêtes
- Triggers pour les timestamps automatiques

## ✅ Validation finale

```sql
-- Script de validation complète
\c futurisys_ml

-- Vérifier les tables
\dt

-- Vérifier les contraintes  
SELECT constraint_name, constraint_type 
FROM information_schema.table_constraints 
WHERE table_schema = 'public';

-- Vérifier les données
SELECT 'employees' as table_name, COUNT(*) as count FROM employees
UNION ALL
SELECT 'model_metadata', COUNT(*) FROM model_metadata;

-- Test d'une vue
SELECT * FROM v_employee_stats_by_dept;
```

**Résultat attendu :**
- 6 tables créées
- 1470 employés importés
- 1 modèle XGBoost enregistré
- Contraintes et index fonctionnels

🎉 **Votre base de données est maintenant prête pour l'intégration avec l'API FastAPI !**