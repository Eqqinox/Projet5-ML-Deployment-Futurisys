#!/usr/bin/env python3
"""
Script d'import du dataset du Projet 4 dans PostgreSQL
Import des 1470 employés avec leurs 27 features + variable cible
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import logging
from datetime import datetime
from dotenv import load_dotenv

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

# Import des modèles depuis l'application
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.database.models import Employee, Base

# Charger les variables d'environnement
load_dotenv()

# Configuration simple
class Settings:
    DATABASE_URL = os.getenv('DATABASE_URL')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_DB = os.getenv('POSTGRES_DB')

settings = Settings()


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_database_url():
    """Crée l'URL de connexion à la base de données"""
    return settings.DATABASE_URL or f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

def create_engine_with_settings(database_url: str = None, echo: bool = False):
    """Crée le moteur SQLAlchemy"""
    if not database_url:
        database_url = create_database_url()
    
    return create_engine(
        database_url,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True
    )

class DatasetImporter:
    """Classe pour importer le dataset du Projet 4 dans PostgreSQL"""

    def __init__(self, database_url: str = None):
        self.engine = create_engine_with_settings(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Création des tables si elles n'existent pas
        Base.metadata.create_all(bind=self.engine)

        self.stats = {
            'total_rows': 0,
            'imported_rows': 0,
            'errors': 0,
            'skipped_rows': 0,
            'start_time': None,
            'end_time': None
        }
    
    def validate_csv_structure(self, df: pd.DataFrame) -> bool:
        """Valide la structure du CSV par rapport au modèle Employee"""
        logger.info("🔍 Validation de la structure du dataset...")
        
        # Colonnes attendues (27 features + target)
        expected_columns = [
            'satisfaction_employee_environnement',
            'note_evaluation_precedente', 
            'niveau_hierarchique_poste',
            'satisfaction_employee_nature_travail',
            'satisfaction_employee_equipe',
            'satisfaction_employee_equilibre_pro_perso',
            'note_evaluation_actuelle',
            'heure_supplementaires',
            'augementation_salaire_precedente',
            'age',
            'genre',
            'revenu_mensuel',
            'statut_marital',
            'departement',
            'poste',
            'nombre_experiences_precedentes',
            'annee_experience_totale',
            'annees_dans_l_entreprise',
            'annees_dans_le_poste_actuel',
            'a_quitte_l_entreprise',
            'nombre_participation_pee',
            'nb_formations_suivies',
            'distance_domicile_travail',
            'niveau_education',
            'domaine_etude',
            'frequence_deplacement',
            'annees_depuis_la_derniere_promotion',
            'annes_sous_responsable_actuel'
        ]
        
        # Vérification des colonnes
        missing_columns = set(expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(expected_columns)
        
        if missing_columns:
            logger.error(f"❌ Colonnes manquantes: {missing_columns}")
            return False
            
        if extra_columns:
            logger.warning(f"⚠️ Colonnes supplémentaires ignorées: {extra_columns}")
        
        # Vérification du nombre de lignes
        if len(df) != 1470:
            logger.warning(f"⚠️ Nombre de lignes attendu: 1470, trouvé: {len(df)}")
        
        # Vérification des valeurs manquantes
        null_counts = df[expected_columns].isnull().sum()
        if null_counts.sum() > 0:
            logger.error("❌ Valeurs manquantes détectées:")
            for col, count in null_counts[null_counts > 0].items():
                logger.error(f"  {col}: {count} valeurs manquantes")
            return False
        
        logger.info("✅ Structure du dataset validée")
        return True
    
    def validate_data_types_and_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide et nettoie les types de données et les plages de valeurs"""
        logger.info("🧹 Validation et nettoyage des données...")
        
        df_clean = df.copy()
        validation_errors = []
        
        # Variables de satisfaction (1-4)
        satisfaction_cols = [
            'satisfaction_employee_environnement',
            'satisfaction_employee_nature_travail', 
            'satisfaction_employee_equipe',
            'satisfaction_employee_equilibre_pro_perso'
        ]
        
        for col in satisfaction_cols:
            mask = (df_clean[col] < 1) | (df_clean[col] > 4)
            if mask.any():
                validation_errors.append(f"{col}: {mask.sum()} valeurs hors plage [1-4]")
        
        # Variables d'évaluation (1-4)
        evaluation_cols = ['note_evaluation_precedente', 'note_evaluation_actuelle']
        for col in evaluation_cols:
            mask = (df_clean[col] < 1) | (df_clean[col] > 4)
            if mask.any():
                validation_errors.append(f"{col}: {mask.sum()} valeurs hors plage [1-4]")
        
        # Niveau hiérarchique (1-5)
        mask = (df_clean['niveau_hierarchique_poste'] < 1) | (df_clean['niveau_hierarchique_poste'] > 5)
        if mask.any():
            validation_errors.append(f"niveau_hierarchique_poste: {mask.sum()} valeurs hors plage [1-5]")
        
        # Âge (18-60)
        mask = (df_clean['age'] < 18) | (df_clean['age'] > 60)
        if mask.any():
            validation_errors.append(f"age: {mask.sum()} valeurs hors plage [18-60]")
        
        # Revenu mensuel (1000-20000)
        mask = (df_clean['revenu_mensuel'] < 1000) | (df_clean['revenu_mensuel'] > 20000)
        if mask.any():
            validation_errors.append(f"revenu_mensuel: {mask.sum()} valeurs hors plage [1000-20000]")
        
        # Variables catégorielles - validation des valeurs
        categorical_validations = {
            'heure_supplementaires': ['Oui', 'Non'],
            'genre': ['F', 'M'],
            'statut_marital': ['Célibataire', 'Marié(e)', 'Divorcé(e)'],
            'departement': ['Commercial', 'Consulting', 'Ressources Humaines'],
            'poste': [
                'Cadre Commercial', 'Assistant de Direction', 'Consultant',
                'Tech Lead', 'Manager', 'Senior Manager', 'Représentant Commercial',
                'Directeur Technique', 'Ressources Humaines'
            ],
            'domaine_etude': [
                'Infra & Cloud', 'Autre', 'Transformation Digitale',
                'Marketing', 'Entrepreunariat', 'Ressources Humaines'
            ],
            'frequence_deplacement': ['Aucun', 'Occasionnel', 'Frequent'],
            'a_quitte_l_entreprise': ['Oui', 'Non']
        }
        
        for col, valid_values in categorical_validations.items():
            invalid_mask = ~df_clean[col].isin(valid_values)
            if invalid_mask.any():
                invalid_values = df_clean[invalid_mask][col].unique()
                validation_errors.append(f"{col}: valeurs invalides {list(invalid_values)}")
        
        # Variables de participation PEE (0-3)
        mask = (df_clean['nombre_participation_pee'] < 0) | (df_clean['nombre_participation_pee'] > 3)
        if mask.any():
            validation_errors.append(f"nombre_participation_pee: {mask.sum()} valeurs hors plage [0-3]")
        
        # Formations suivies (0-6)
        mask = (df_clean['nb_formations_suivies'] < 0) | (df_clean['nb_formations_suivies'] > 6)
        if mask.any():
            validation_errors.append(f"nb_formations_suivies: {mask.sum()} valeurs hors plage [0-6]")
        
        # Distance domicile-travail (1-29)
        mask = (df_clean['distance_domicile_travail'] < 1) | (df_clean['distance_domicile_travail'] > 29)
        if mask.any():
            validation_errors.append(f"distance_domicile_travail: {mask.sum()} valeurs hors plage [1-29]")
        
        # Niveau éducation (1-5)
        mask = (df_clean['niveau_education'] < 1) | (df_clean['niveau_education'] > 5)
        if mask.any():
            validation_errors.append(f"niveau_education: {mask.sum()} valeurs hors plage [1-5]")
        
        if validation_errors:
            logger.warning("⚠️ Erreurs de validation détectées (affichage seulement):")
            for error in validation_errors:
                logger.warning(f"  {error}")
            # Ne pas lever d'erreur, juste afficher les warnings
        
        logger.info("✅ Validation des données terminée")
        return df_clean
    
    def prepare_data_for_import(self, df: pd.DataFrame) -> List[Dict]:
        """Prépare les données pour l'insertion en base"""
        logger.info("🔧 Préparation des données pour l'import...")
        
        records = []
        for _, row in df.iterrows():
            record = row.to_dict()
            record['created_at'] = datetime.now()
            record['updated_at'] = datetime.now()
            records.append(record)
        
        logger.info(f"✅ {len(records)} enregistrements préparés")
        return records
    
    def clear_existing_data(self, session, confirm: bool = False) -> bool:
        """Supprime les données existantes"""
        existing_count = session.query(Employee).count()
        
        if existing_count == 0:
            logger.info("📝 Aucune donnée existante à supprimer")
            return True
        
        logger.warning(f"⚠️ {existing_count} employés trouvés dans la base")
        
        if not confirm:
            response = input("❓ Supprimer les données existantes ? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                logger.info("❌ Import annulé par l'utilisateur")
                return False
        
        try:
            deleted_count = session.query(Employee).delete()
            session.commit()
            logger.info(f"🗑️ {deleted_count} enregistrements supprimés")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur lors de la suppression: {e}")
            session.rollback()
            return False
    
    def import_data_batch(self, session, records: List[Dict], batch_size: int = 100):
        """Importe les données par batch"""
        logger.info(f"📥 Import de {len(records)} enregistrements par batch de {batch_size}...")
        
        imported_count = 0
        error_count = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            try:
                employees = [Employee(**record) for record in batch]
                session.add_all(employees)
                session.commit()
                
                imported_count += len(batch)
                logger.info(f"✅ Batch {i//batch_size + 1}: {len(batch)} enregistrements importés")
                
            except Exception as e:
                logger.error(f"❌ Erreur batch {i//batch_size + 1}: {e}")
                session.rollback()
                error_count += len(batch)
                
                # Tentative d'import ligne par ligne
                for j, record in enumerate(batch):
                    try:
                        employee = Employee(**record)
                        session.add(employee)
                        session.commit()
                        imported_count += 1
                    except Exception as detail_error:
                        logger.error(f"❌ Erreur ligne {i+j+1}: {detail_error}")
                        session.rollback()
        
        self.stats['imported_rows'] = imported_count
        self.stats['errors'] = error_count
        
        logger.info(f"📊 Import terminé: {imported_count} réussites, {error_count} erreurs")
        return imported_count, error_count
    
    def verify_import(self, session):
        """Vérifie l'intégrité des données importées"""
        logger.info("🔍 Vérification de l'import...")
        
        total_count = session.query(Employee).count()
        logger.info(f"📊 Total employés en base: {total_count}")
        
        # Statistiques par département
        dept_stats = session.query(
            Employee.departement,
            func.count(Employee.employee_id).label('count'),
            func.count(func.nullif(Employee.a_quitte_l_entreprise, 'Non')).label('quit_count')
        ).group_by(Employee.departement).all()
        
        logger.info("📈 Statistiques par département:")
        for dept, count, quit_count in dept_stats:
            quit_rate = (quit_count / count * 100) if count > 0 else 0
            logger.info(f"  {dept}: {count} employés, {quit_count} démissions ({quit_rate:.1f}%)")
        
        logger.info("✅ Vérification terminée")
        return True
    
    def import_dataset(self, csv_file_path: str, clear_existing: bool = False, 
                      batch_size: int = 100, confirm_clear: bool = False) -> Dict:
        """Fonction principale d'import du dataset"""
        self.stats['start_time'] = datetime.now()
        
        try:
            logger.info("🚀 Début de l'import du dataset Projet 4")
            logger.info("=" * 60)
            
            # Chargement du CSV
            logger.info(f"📖 Chargement du fichier: {csv_file_path}")
            
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_file_path}")
            
            df = pd.read_csv(csv_file_path)
            self.stats['total_rows'] = len(df)
            
            logger.info(f"📊 Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Validation de la structure
            if not self.validate_csv_structure(df):
                raise ValueError("Structure du CSV invalide")
            
            # Validation et nettoyage des données
            df_clean = self.validate_data_types_and_ranges(df)
            
            # Préparation des données
            records = self.prepare_data_for_import(df_clean)
            
            # Connexion à la base
            session = self.SessionLocal()
            
            try:
                # Suppression des données existantes (si demandé)
                if clear_existing:
                    if not self.clear_existing_data(session, confirm_clear):
                        raise Exception("Suppression des données annulée")
                
                # Import des données
                imported_count, error_count = self.import_data_batch(session, records, batch_size)
                
                # Vérification de l'import
                self.verify_import(session)
                
                self.stats['end_time'] = datetime.now()
                duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
                
                logger.info("=" * 60)
                logger.info("✅ Import terminé avec succès !")
                logger.info(f"📊 {imported_count}/{self.stats['total_rows']} enregistrements importés")
                logger.info(f"⏱️ Durée: {duration:.2f} secondes")
                
                return {
                    'success': True,
                    'imported_count': imported_count,
                    'total_count': self.stats['total_rows'],
                    'duration_seconds': duration
                }
                
            finally:
                session.close()
                
        except Exception as e:
            self.stats['end_time'] = datetime.now()
            logger.error(f"💥 Erreur lors de l'import: {e}")
            raise e

def find_dataset_file() -> Optional[str]:
    """Recherche automatique du fichier dataset"""
    possible_paths = [
        "data/raw/dataset_projet4.csv",
        "data/dataset_projet4.csv", 
        "dataset_projet4.csv",
        "../dataset_projet4.csv",
        "../../dataset_projet4.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import du dataset Projet 4 dans PostgreSQL")
    parser.add_argument("--file", type=str, help="Chemin vers le fichier CSV")
    parser.add_argument("--clear", action="store_true", help="Supprimer les données existantes")
    parser.add_argument("--batch-size", type=int, default=100, help="Taille des batches")
    parser.add_argument("--yes", action="store_true", help="Confirmer automatiquement")
    parser.add_argument("--url", type=str, help="URL de connexion personnalisée")
    
    args = parser.parse_args()
    
    try:
        # Recherche du fichier CSV
        csv_file = args.file
        if not csv_file:
            csv_file = find_dataset_file()
            if not csv_file:
                logger.error("❌ Fichier CSV non trouvé. Utilisez --file pour spécifier le chemin")
                sys.exit(1)
            else:
                logger.info(f"📁 Fichier trouvé automatiquement: {csv_file}")
        
        # Import du dataset
        importer = DatasetImporter(database_url=args.url)
        
        result = importer.import_dataset(
            csv_file_path=csv_file,
            clear_existing=args.clear,
            batch_size=args.batch_size,
            confirm_clear=args.yes
        )
        
        logger.info("🎉 Script d'import terminé avec succès !")
        
    except Exception as e:
        logger.error(f"💥 Échec de l'import: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()