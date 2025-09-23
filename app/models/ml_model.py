"""
Wrapper pour le modèle XGBoost de prédiction d'attrition
Gestion du chargement, des prédictions et de l'explicabilité
"""

import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
import logging
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from app.models.schemas import EmployeeData, PredictionResult
from app.core.config import settings

logger = logging.getLogger(__name__)

class MLModel:
    """
    Wrapper pour le modèle XGBoost de prédiction d'attrition des employés
    Adaptation de l'encodage OneHot utilisé dans le Projet 4
    """
    
    def __init__(self):
        self.model = None
        self.ml_model_version = settings.MODEL_VERSION
        self.threshold = 0.514  # Seuil optimal du Model
        
        # Encodeurs (chargés depuis les fichiers sauvegardés du Projet 4)
        self.onehot_encoder = None
        self.ordinal_encoder = None
        self.final_column_names = None
        self.model_info = None
        
        # Variables pour l'encodage
        self.variables_catego = ['departement', 'domaine_etude', 'poste', 'statut_marital']
        self.is_loaded = False
        
    def _load_encoders(self):
        """
        Charge les encodeurs sauvegardés depuis le Projet 4
        """
        try:
            models_dir = Path(settings.MODEL_PATH).parent
            
            # Chargement des encodeurs
            onehot_path = models_dir / 'onehot_encoder.pkl'
            ordinal_path = models_dir / 'ordinal_encoder.pkl'
            columns_path = models_dir / 'final_column_names.pkl'
            info_path = models_dir / 'model_info.pkl'
            
            if not all([onehot_path.exists(), ordinal_path.exists(), columns_path.exists()]):
                raise FileNotFoundError(
                    f"Encodeurs manquants dans {models_dir}. "
                    f"Exécutez d'abord le script de sauvegarde dans le Projet 4."
                )
            
            self.onehot_encoder = joblib.load(onehot_path)
            self.ordinal_encoder = joblib.load(ordinal_path)
            self.final_column_names = joblib.load(columns_path)
            
            if info_path.exists():
                self.model_info = joblib.load(info_path)
            
            logger.info(f"✅ Encodeurs chargés avec succès")
            logger.info(f"📊 Nombre de features finales: {len(self.final_column_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des encodeurs: {e}")
            raise e
        
    def load_model(self) -> bool:
        """
        Charge le modèle XGBoost et les encodeurs depuis les fichiers .pkl
        """
        try:
            model_path = Path(settings.MODEL_PATH)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
            
            # Chargement du modèle XGBoost
            self.model = joblib.load(model_path)
            
            # Chargement des encodeurs
            self._load_encoders()
            
            self.is_loaded = True
            
            logger.info(f"✅ Modèle chargé avec succès depuis {model_path}")
            logger.info(f"📊 Type de modèle: {type(self.model)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise e
    
    def _preprocess_employee_data(self, employee: EmployeeData) -> np.ndarray:
        """
        Préprocesse les données d'un employé selon l'encodage du Projet 4
        """
        try:
            # Conversion en dictionnaire
            data_dict = employee.dict()
            
            # Créer un DataFrame avec une seule ligne
            df = pd.DataFrame([data_dict])
            
            # Encodage binaire pour heure_supplementaires
            df.loc[df['heure_supplementaires'] == 'Non', 'heure_supplementaires'] = 0
            df.loc[df['heure_supplementaires'] == 'Oui', 'heure_supplementaires'] = 1
            df['heure_supplementaires'] = df['heure_supplementaires'].astype(int)
            
            # Encodage binaire pour genre (adapter selon vos données d'entrée)
            # Note: Votre schéma Pydantic utilise "Homme"/"Femme" mais votre code "M"/"F"
            # Adaptation nécessaire:
            df.loc[df['genre'] == 'Femme', 'genre'] = 0
            df.loc[df['genre'] == 'Homme', 'genre'] = 1
            df['genre'] = df['genre'].astype(int)
            
            # Augementation_salaire_precedente reste float (c'est déjà un pourcentage)
            # Pas de transformation nécessaire
            
            # Encodage ordinal pour frequence_deplacement
            # Adapter les valeurs du schéma vers les valeurs d'entraînement
            freq_mapping = {
                'Pas_de_Voyage': 'Aucun',
                'Voyage_Rare': 'Occasionnel', 
                'Voyage_Fréquent': 'Frequent'
            }
            df['frequence_deplacement'] = df['frequence_deplacement'].map(freq_mapping)
            df[['frequence_deplacement']] = self.ordinal_encoder.transform(df[['frequence_deplacement']])
            
            # Encodage OneHot pour les variables nominales
            df_categorical = df[self.variables_catego]
            df_encoded = self.onehot_encoder.transform(df_categorical)
            
            # Créer DataFrame avec les nouvelles colonnes
            feature_names = self.onehot_encoder.get_feature_names_out(self.variables_catego)
            df_onehot = pd.DataFrame(df_encoded, columns=feature_names, index=df.index)
            
            # Supprimer les colonnes originales et concatener les encodées
            df = df.drop(self.variables_catego, axis=1)
            df = pd.concat([df, df_onehot], axis=1)
            
            # Supprimer la variable cible si elle existe
            if 'a_quitte_l_entreprise' in df.columns:
                df = df.drop('a_quitte_l_entreprise', axis=1)
            
            # Réorganiser les colonnes selon l'ordre d'entraînement
            if self.final_column_names:
                # S'assurer que toutes les colonnes attendues sont présentes
                missing_cols = set(self.final_column_names) - set(df.columns)
                if missing_cols:
                    logger.error(f"Colonnes manquantes après encodage: {missing_cols}")
                    raise ValueError(f"Colonnes manquantes: {missing_cols}")
                
                # Réorganiser dans le bon ordre
                df = df[self.final_column_names]
            
            # Conversion en array NumPy
            return df.values
            
        except Exception as e:
            logger.error(f"Erreur lors du préprocessing: {e}")
            logger.error(f"Colonnes disponibles: {df.columns.tolist() if 'df' in locals() else 'DataFrame non créé'}")
            if hasattr(self, 'final_column_names') and self.final_column_names:
                logger.error(f"Colonnes attendues: {self.final_column_names[:10]}...")
            raise e
    
    def predict_single(self, employee: EmployeeData, employee_id: Optional[str] = None) -> PredictionResult:
        """
        Effectue une prédiction pour un seul employé
        """
        if not self.is_loaded:
            raise RuntimeError("Le modèle n'est pas chargé")
        
        try:
            # Préprocessing
            X = self._preprocess_employee_data(employee)
            
            # Prédiction des probabilités
            probabilities = self.model.predict_proba(X)[0]
            prob_stay = probabilities[0]  # Probabilité de rester (classe 0)
            prob_quit = probabilities[1]  # Probabilité de partir (classe 1)
            
            # Décision basée sur le seuil
            prediction = "Oui" if prob_quit >= self.threshold else "Non"
            
            # Niveau de confiance
            confidence_level = self._get_confidence_level(max(prob_stay, prob_quit))
            
            # Facteurs de risque (basique, peut être amélioré avec SHAP)
            risk_factors = self._identify_risk_factors(employee)
            
            return PredictionResult(
                employee_id=employee_id,
                prediction=prediction,
                probability_quit=round(prob_quit, 4),
                probability_stay=round(prob_stay, 4),
                confidence_level=confidence_level,
                risk_factors=risk_factors,
                model_version=self.ml_model_version,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise e
    
    def predict_batch(self, employees: List[EmployeeData]) -> List[PredictionResult]:
        """
        Effectue des prédictions pour plusieurs employés
        """
        if not self.is_loaded:
            raise RuntimeError("Le modèle n'est pas chargé")
        
        predictions = []
        for i, employee in enumerate(employees):
            try:
                prediction = self.predict_single(employee, employee_id=f"emp_{i+1}")
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Erreur pour l'employé {i+1}: {e}")
                # Continuer avec les autres employés
                continue
        
        return predictions
    
    def _get_confidence_level(self, max_probability: float) -> str:
        """
        Détermine le niveau de confiance basé sur la probabilité maximale
        """
        if max_probability >= 0.8:
            return "Élevé"
        elif max_probability >= 0.6:
            return "Moyen"
        else:
            return "Faible"
    
    def _identify_risk_factors(self, employee: EmployeeData) -> List[str]:
        """
        Identifie les facteurs de risque basiques
        """
        risk_factors = []
        
        # Règles heuristiques basiques
        if employee.satisfaction_employee_environnement <= 3:
            risk_factors.append("Satisfaction environnement très faible")
        
        if employee.satisfaction_employee_nature_travail <= 3:
            risk_factors.append("Satisfaction travail très faible")
        
        if employee.heure_supplementaires == "Oui":
            risk_factors.append("Heures supplémentaires fréquentes")
        
        if employee.augementation_salaire_precedente == "Non":
            risk_factors.append("Pas d'augmentation récente")
        
        if employee.annees_depuis_la_derniere_promotion >= 5:
            risk_factors.append("Pas de promotion depuis longtemps")
        
        if employee.distance_domicile_travail >= 30:
            risk_factors.append("Distance domicile-travail importante")
        
        return risk_factors[:5]  # Limiter à 5 facteurs principaux
    
    def get_model_info(self) -> Dict:
        """
        Retourne les informations sur le modèle
        """
        
        base_info = {
            "model_name": "XGBoost Employee Attrition Classifier",
            "model_type": "XGBoost Classifier", 
            "version": self.ml_model_version,
            "threshold": self.threshold,
            "is_loaded": self.is_loaded,
            "encoding": "OneHot pour variables nominales, Ordinal pour fréquence",
        }
        
        # Ajouter les informations du Projet 4
        if self.model_info:
            base_info.update({
                "original_features": self.model_info.get('original_features_count', 27),
                "encoded_features": self.model_info.get('final_features_count', 'Unknown'),
                "categorical_variables_encoded": self.model_info.get('categorical_variables_encoded', []),
            })
        elif self.final_column_names:
            base_info.update({
                "original_features": 27,
                "encoded_features": len(self.final_column_names),
                "categorical_variables_encoded": self.variables_catego,
            })
        
        # Métriques réelles du modèle avec validation croisée stratifiée
        base_info["performance_metrics"] = {
            "accuracy": 0.8588,          # CV moyenne
            "accuracy_std": 0.0220,      # Écart-type CV
            "precision": 0.5654,         # CV moyenne  
            "precision_std": 0.0701,     # Écart-type CV
            "recall": 0.5684,            # CV moyenne
            "recall_std": 0.0678,        # Écart-type CV
            "f1_score": 0.5656,          # CV moyenne
            "f1_std": 0.0638,            # Écart-type CV
            "roc_auc": 0.8252,           # CV moyenne
            "roc_auc_std": 0.0212        # Écart-type CV
        }
        
        return base_info
    
    def health_check(self) -> Dict[str, bool]:
        """
        Vérifie la santé du modèle
        """
        return {
            "model_loaded": self.is_loaded,
            "model_file_exists": os.path.exists(settings.MODEL_PATH),
            "encoders_loaded": self.onehot_encoder is not None and self.ordinal_encoder is not None,
            "column_names_loaded": self.final_column_names is not None,
            "all_encoder_files_exist": self._check_encoder_files_exist()
        }
        
    def _check_encoder_files_exist(self) -> bool:
        """Vérifie que tous les fichiers d'encodeurs existent"""
        try:
            models_dir = Path(settings.MODEL_PATH).parent
            required_files = [
                'onehot_encoder.pkl',
                'ordinal_encoder.pkl', 
                'final_column_names.pkl'
            ]
            return all((models_dir / f).exists() for f in required_files)
        except:
            return False