# Sprint 1 - S1-T5 : Identifier et expliquer les variables
import pandas as pd
from pathlib import Path


def racine_projet():
    """Remonte les dossiers jusqu'à trouver le dossier 'data' (racine du projet)."""
    dossier = Path(__file__).resolve().parent
    for p in [dossier, *dossier.parents]:
        if (p / 'data').is_dir():
            return p
    return dossier


BASE = racine_projet()
(BASE / 'resultats').mkdir(exist_ok=True)

# Chargement du dataset brut (séparateur ';')
csv_brut = BASE / 'data' / 'cardio_data.csv'
if not csv_brut.exists():
    csv_brut = BASE / 'data' / 'cardio_dataset.csv'
df = pd.read_csv(csv_brut, sep=';')

descriptions = {
    'id': "Identifiant du patient (inutile pour la prédiction)",
    'age': "Âge en jours",
    'gender': "Sexe : 1 = femme, 2 = homme",
    'height': "Taille en cm",
    'weight': "Poids en kg",
    'ap_hi': "Tension artérielle systolique (mmHg)",
    'ap_lo': "Tension artérielle diastolique (mmHg)",
    'cholesterol': "Cholestérol : 1 = normal, 2 = élevé, 3 = très élevé",
    'gluc': "Glucose : 1 = normal, 2 = élevé, 3 = très élevé",
    'smoke': "Fumeur : 0 = non, 1 = oui",
    'alco': "Consomme de l'alcool : 0 = non, 1 = oui",
    'active': "Activité physique : 0 = non, 1 = oui",
    'cardio': "CIBLE : maladie cardiaque (0 = non, 1 = oui)",
}

print(f"{'Variable':<13}{'Type':<9}{'Valeurs uniques':<17}Signification")
print("-" * 90)
for col in df.columns:
    print(f"{col:<13}{str(df[col].dtype):<9}{df[col].nunique():<17}{descriptions.get(col, '')}")

print("\nVariable cible :", "cardio")
print("Variables explicatives :", [c for c in df.columns if c not in ('id', 'cardio')])
