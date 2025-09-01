"""
Schémas Pydantic pour la validation des données d'entrée et sortie
Modèle de prédiction d'attrition des employés - Futurisys
"""    
schema_extra = {
    "example":{
        "satisfaction_employee_environnement": 7,
        "satisfaction_employee_nature_travail": 8,
        "satisfaction_employee_equipe": 6,
        "satisfaction_employee_equilibre_pro_perso": 7,
        "note_evaluation_precedente": 4,
        "note_evaluation_actuelle": 4,
        "niveau_hierarchique_poste": 2,
        "heure_supplementaires": "Oui",
        "augementation_salaire_precedente": 0.15,  # Pourcentage
        "age": 32,
        "genre": "Homme",
        "revenu_mensuel": 3500,
        "statut_marital": "Marié(e)",  # Corrected value
        "departement": "Commercial",
        "poste": "Manager",
        "nombre_experiences_precedentes": 2,
        "annee_experience_totale": 8,
        "annees_dans_l_entreprise": 3,
        "annees_dans_le_poste_actuel": 2,
    }
}


from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
from datetime import datetime
from enum import Enum

# Énumérations pour les variables catégorielles (adaptées à vos données exactes)
class HeuresSupplementaires(str, Enum):
    OUI = "Oui"
    NON = "Non"

class Genre(str, Enum):
    HOMME = "Homme"
    FEMME = "Femme"

class StatutMarital(str, Enum):
    CELIBATAIRE = "Célibataire" 
    MARIE = "Marié(e)"  # Changé pour correspondre à vos données d'entraînement
    DIVORCE = "Divorcé(e)"  # Changé pour correspondre à vos données d'entraînement

class Departement(str, Enum):
    COMMERCIAL = "Commercial"
    CONSULTING = "Consulting" 
    RESSOURCES_HUMAINES = "Ressources Humaines"

class Poste(str, Enum):
    CADRE_COMMERCIAL = "Cadre Commercial"
    ASSISTANT_DIRECTION = "Assistant de Direction"
    CONSULTANT = "Consultant"
    TECH_LEAD = "Tech Lead"
    MANAGER = "Manager"
    SENIOR_MANAGER = "Senior Manager"
    REPRESENTANT_COMMERCIAL = "Représentant Commercial"
    DIRECTEUR_TECHNIQUE = "Directeur Technique"
    RESSOURCES_HUMAINES = "Ressources Humaines"
    
class Config:
    json_schema_extra = {  # Remplace schema_extra
        "example": {...}
    }
    
class DomaineEtude(str, Enum):
    INFRA_CLOUD = "Infra & Cloud"
    AUTRE = "Autre"
    TRANSFORMATION_DIGITALE = "Transformation Digitale"
    MARKETING = "Marketing"
    ENTREPRENEURIAT = "Entrepreneuriat"
    RESSOURCES_HUMAINES = "Ressources Humaines"

class FrequenceDeplacement(str, Enum):
    VOYAGE_FREQUENT = "Voyage_Fréquent"  # Mappé vers "Frequent"
    VOYAGE_RARE = "Voyage_Rare"          # Mappé vers "Occasionnel"
    PAS_DE_VOYAGE = "Pas_de_Voyage"      # Mappé vers "Aucun"

# Modèle d'entrée pour une prédiction unique
class EmployeeData(BaseModel):
    """
    Données d'un employé pour la prédiction d'attrition
    """
    # Variables de satisfaction (1-4, pas 1-10 !)
    satisfaction_employee_environnement: int = Field(
        ..., ge=1, le=4,
        description="Satisfaction de l'employé vis-à-vis de l'environnement de travail (1-4)"
    )
    satisfaction_employee_nature_travail: int = Field(
        ..., ge=1, le=4,
        description="Satisfaction de l'employé vis-à-vis de la nature du travail (1-4)"
    )
    satisfaction_employee_equipe: int = Field(
        ..., ge=1, le=4,
        description="Satisfaction de l'employé vis-à-vis de son équipe (1-4)"
    )
    satisfaction_employee_equilibre_pro_perso: int = Field(
        ..., ge=1, le=4,
        description="Satisfaction de l'employé vis-à-vis de l'équilibre vie pro/perso (1-4)"
    )
    
    # Variables d'évaluation (1-4, pas 1-5 !)
    note_evaluation_precedente: int = Field(
        ..., ge=1, le=4,
        description="Note de l'évaluation précédente (1-4)"
    )
    note_evaluation_actuelle: int = Field(
        ..., ge=1, le=4,
        description="Note de l'évaluation actuelle (1-4)"
    )
    
    # Variables hiérarchiques et organisationnelles
    niveau_hierarchique_poste: int = Field(
        ..., ge=1, le=5,
        description="Niveau hiérarchique du poste (1-5)"
    )
    
    # Variables binaires
    heure_supplementaires: HeuresSupplementaires = Field(
        ..., description="L'employé fait-il des heures supplémentaires?"
    )
    
    # Variable d'augmentation (maintenant un float - pourcentage)
    augementation_salaire_precedente: float = Field(
        ..., ge=0.11, le=0.25,
        description="Pourcentage d'augmentation salariale précédente (0.11 à 0.25)"
    )
    
    # Variables démographiques
    age: int = Field(
        ..., ge=18, le=65,
        description="Âge de l'employé (18-65 ans)"
    )
    genre: Genre = Field(
        ..., description="Genre de l'employé"
    )
    revenu_mensuel: int = Field(
        ..., ge=1000, le=50000,
        description="Revenu mensuel de l'employé (en euros)"
    )
    statut_marital: StatutMarital = Field(
        ..., description="Statut marital de l'employé"
    )
    
    # Variables organisationnelles (avec vos valeurs exactes)
    departement: Departement = Field(
        ..., description="Département de l'employé"
    )
    poste: Poste = Field(
        ..., description="Poste occupé par l'employé"
    )
    
    # Variables d'expérience
    nombre_experiences_precedentes: int = Field(
        ..., ge=0, le=20,
        description="Nombre d'expériences professionnelles précédentes"
    )
    annee_experience_totale: int = Field(
        ..., ge=0, le=40,
        description="Nombre total d'années d'expérience"
    )
    annees_dans_l_entreprise: int = Field(
        ..., ge=0, le=30,
        description="Nombre d'années dans l'entreprise actuelle"
    )
    annees_dans_le_poste_actuel: int = Field(
        ..., ge=0, le=20,
        description="Nombre d'années dans le poste actuel"
    )
    annees_depuis_la_derniere_promotion: int = Field(
        ..., ge=0, le=15,
        description="Nombre d'années depuis la dernière promotion"
    )
    annes_sous_responsable_actuel: int = Field(
        ..., ge=0, le=15,
        description="Nombre d'années sous le responsable actuel"
    )
    
    # Variables de formation et développement (plages correctes)
    nombre_participation_pee: int = Field(
        ..., ge=0, le=3,
        description="Nombre de participations au plan d'épargne entreprise (0-3)"
    )
    nb_formations_suivies: int = Field(
        ..., ge=0, le=6,
        description="Nombre de formations suivies (0-6)"
    )
    
    # Variables géographiques et logistiques
    distance_domicile_travail: int = Field(
        ..., ge=0, le=100,
        description="Distance entre domicile et travail (en km)"
    )
    
    # Variables d'éducation (avec vos valeurs exactes)
    niveau_education: int = Field(
        ..., ge=1, le=5,
        description="Niveau d'éducation (1-5)"
    )
    domaine_etude: DomaineEtude = Field(
        ..., description="Domaine d'étude de l'employé"
    )
    
    # Fréquence de déplacement
    frequence_deplacement: FrequenceDeplacement = Field(
        ..., description="Fréquence des déplacements professionnels"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "satisfaction_employee_environnement": 3,  # 1-4
                "satisfaction_employee_nature_travail": 4,  # 1-4
                "satisfaction_employee_equipe": 2,          # 1-4
                "satisfaction_employee_equilibre_pro_perso": 3,  # 1-4
                "note_evaluation_precedente": 3,            # 1-4
                "note_evaluation_actuelle": 3,              # 1-4
                "niveau_hierarchique_poste": 2,             # 1-5
                "heure_supplementaires": "Oui",
                "augementation_salaire_precedente": 0.15,   # 0.11-0.25
                "age": 32,
                "genre": "Homme",
                "revenu_mensuel": 3500,
                "statut_marital": "Marié",
                "departement": "Commercial",
                "poste": "Manager",
                "nombre_experiences_precedentes": 2,
                "annee_experience_totale": 8,
                "annees_dans_l_entreprise": 3,
                "annees_dans_le_poste_actuel": 2,
                "annees_depuis_la_derniere_promotion": 1,
                "annes_sous_responsable_actuel": 2,
                "nombre_participation_pee": 1,              # 0-3
                "nb_formations_suivies": 3,                 # 0-6
                "distance_domicile_travail": 15,
                "niveau_education": 4,                      # 1-5
                "domaine_etude": "Marketing",
                "frequence_deplacement": "Voyage_Rare"
            }
        }

# Modèle pour les prédictions batch
class BatchEmployeeData(BaseModel):
    """
    Liste d'employés pour les prédictions batch
    """
    employees: List[EmployeeData] = Field(
        ..., min_items=1, max_items=100,
        description="Liste des employés (maximum 100)"
    )

# Modèles de sortie
class PredictionResult(BaseModel):
    """
    Résultat de prédiction pour un employé
    """
    employee_id: Optional[str] = Field(None, description="Identifiant de l'employé (optionnel)")
    prediction: str = Field(..., description="Prédiction: 'Oui' ou 'Non' pour l'attrition")
    probability_quit: float = Field(..., description="Probabilité que l'employé quitte (0-1)")
    probability_stay: float = Field(..., description="Probabilité que l'employé reste (0-1)")
    confidence_level: str = Field(..., description="Niveau de confiance: 'Élevé', 'Moyen', 'Faible'")
    risk_factors: List[str] = Field(default_factory=list, description="Principaux facteurs de risque identifiés")
    model_version: str = Field(..., description="Version du modèle utilisé")
    timestamp: datetime = Field(default_factory=datetime.now, description="Horodatage de la prédiction")

class BatchPredictionResult(BaseModel):
    """
    Résultat de prédictions batch
    """
    predictions: List[PredictionResult] = Field(..., description="Liste des prédictions")
    total_employees: int = Field(..., description="Nombre total d'employés traités")
    quit_predictions: int = Field(..., description="Nombre de prédictions 'Oui' (attrition)")
    stay_predictions: int = Field(..., description="Nombre de prédictions 'Non' (rétention)")
    average_quit_probability: float = Field(..., description="Probabilité moyenne d'attrition")
    processing_time_seconds: float = Field(..., description="Temps de traitement en secondes")

class ModelInfo(BaseModel):
    """
    Informations sur le modèle de ML
    """
    model_name: str = Field(..., description="Nom du modèle")
    model_type: str = Field(..., description="Type d'algorithme")
    version: str = Field(..., description="Version du modèle")
    features_count: int = Field(..., description="Nombre de features")
    training_date: Optional[str] = Field(None, description="Date d'entraînement")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Métriques de performance")
    threshold: float = Field(..., description="Seuil de décision optimisé")

# Modèles pour la gestion d'erreurs
class ErrorResponse(BaseModel):
    """
    Réponse d'erreur standardisée
    """
    error: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur détaillé")
    details: Optional[Dict] = Field(None, description="Détails supplémentaires")
    timestamp: datetime = Field(default_factory=datetime.now, description="Horodatage de l'erreur")