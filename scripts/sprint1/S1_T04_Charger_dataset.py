# Sprint 1 - S1-T4 : Charger le dataset (Pandas) et examiner la structure
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

print(f"Fichier chargé : {csv_brut.name}")
print(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")

print("\n--- Premières lignes ---")
print(df.head())

print("\n--- Informations générales ---")
df.info()

print("\n--- Répartition de la cible (cardio) ---")
print(df['cardio'].value_counts())
