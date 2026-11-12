# Sprint 1 - S1-T8 : Traiter les manquants et supprimer les doublons
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

print(f"Dataset brut : {len(df)} lignes")

# 1. Suppression de l'identifiant (inutile pour la prédiction)
df_propre = df.drop(columns=['id'])

# 2. Valeurs manquantes
n_manquantes = int(df_propre.isnull().sum().sum())
df_propre = df_propre.dropna()
print(f"\n1. Valeurs manquantes traitées : {n_manquantes} (aucune ligne supprimée si 0)")

# 3. Doublons exacts
n_avant = len(df_propre)
df_propre = df_propre.drop_duplicates()
print(f"2. Doublons supprimés : {n_avant - len(df_propre)}")
print(f"   Doublons restants  : {int(df_propre.duplicated().sum())}")

# 4. Valeurs aberrantes de tension (étape de nettoyage complémentaire)
n_avant = len(df_propre)
df_propre = df_propre[(df_propre['ap_hi'] >= 50) & (df_propre['ap_hi'] <= 250) &
                      (df_propre['ap_lo'] >= 30) & (df_propre['ap_lo'] <= 200) &
                      (df_propre['ap_hi'] >= df_propre['ap_lo'])].copy()
print(f"3. Lignes avec tension aberrante supprimées : {n_avant - len(df_propre)}")

# 5. Âge en années (plus lisible que des jours)
df_propre['age_years'] = (df_propre['age'] / 365.25).astype(int)
df_propre = df_propre.drop(columns=['age'])
print("4. Colonne 'age' (jours) remplacée par 'age_years' (années)")

print(f"\nDataset nettoyé : {df_propre.shape[0]} lignes, {df_propre.shape[1]} colonnes")
print(f"Lignes supprimées au total : {len(df) - len(df_propre)}")
print(df_propre.head())
