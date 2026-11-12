# Sprint 1 - S1-T6 : Analyser les types et statistiques
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

print("--- Types de données ---")
print(df.dtypes)

print("\n--- Statistiques descriptives ---")
print(df.describe().T.round(2))

print("\n--- Valeurs minimales et maximales suspectes ---")
for col in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
    print(f"{col:<8} min = {df[col].min():<8} max = {df[col].max()}")
