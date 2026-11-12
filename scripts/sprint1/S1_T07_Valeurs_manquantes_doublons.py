# Sprint 1 - S1-T7 : Détecter les valeurs manquantes et doublons
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

print("--- Valeurs manquantes par colonne ---")
print(df.isnull().sum())
print(f"Total de valeurs manquantes : {df.isnull().sum().sum()}")

# On ignore 'id' : deux lignes identiques avec des id différents sont des doublons
doublons = df.drop(columns=['id']).duplicated().sum()
print(f"\nNombre de lignes dupliquées (hors id) : {doublons}")

print("\n--- Valeurs incohérentes (tension) ---")
print(f"ap_hi hors de [50, 250]   : {((df['ap_hi'] < 50) | (df['ap_hi'] > 250)).sum()}")
print(f"ap_lo hors de [30, 200]   : {((df['ap_lo'] < 30) | (df['ap_lo'] > 200)).sum()}")
print(f"ap_hi inférieur à ap_lo   : {(df['ap_hi'] < df['ap_lo']).sum()}")
