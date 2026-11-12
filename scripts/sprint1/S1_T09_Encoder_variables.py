# Sprint 1 - S1-T9 : Encoder les variables catégorielles
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


def nettoyer(df):
    """Nettoyage : doublons, valeurs manquantes, tensions aberrantes, âge en années."""
    df = df.drop(columns=['id']).drop_duplicates().dropna()
    df = df[(df['ap_hi'] >= 50) & (df['ap_hi'] <= 250) &
            (df['ap_lo'] >= 30) & (df['ap_lo'] <= 200) &
            (df['ap_hi'] >= df['ap_lo'])].copy()
    df['age_years'] = (df['age'] / 365.25).astype(int)
    return df.drop(columns=['age'])


def encoder(df):
    """Encodage : gender 1/2 -> 0/1 (0 = femme, 1 = homme). cholesterol et gluc
    sont ordinaux (1 < 2 < 3) et smoke / alco / active / cardio sont déjà 0/1."""
    df = df.copy()
    df['gender'] = df['gender'].map({1: 0, 2: 1})
    return df


df_propre = nettoyer(df)
df_enc = encoder(df_propre)

print("--- Avant encodage : gender ---")
print(df_propre['gender'].value_counts())
print("\n--- Après encodage : gender (0 = femme, 1 = homme) ---")
print(df_enc['gender'].value_counts())

print("\nVariables ordinales conservées telles quelles :")
print("  cholesterol :", sorted(df_enc['cholesterol'].unique().tolist()))
print("  gluc        :", sorted(df_enc['gluc'].unique().tolist()))
print("Variables déjà binaires (0/1) : smoke, alco, active")
print(df_enc.head())
