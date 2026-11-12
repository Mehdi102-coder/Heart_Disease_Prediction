# Sprint 1 - S1-T11 : Découpage Train / Test (X et y)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


df_enc = encoder(nettoyer(df))

# 1. Séparation des variables explicatives (X) et de la cible (y)
X = df_enc.drop(columns=['cardio'])
y = df_enc['cardio']

# 2. Découpage : 80 % entraînement, 20 % test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalisation : fit sur le train UNIQUEMENT, puis transform sur le test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X :", X.shape, "| y :", y.shape)
print("X_train :", X_train_scaled.shape, "| X_test :", X_test_scaled.shape)
print("\nPart de malades dans y_train : {:.1%}".format(y_train.mean()))
print("Part de malades dans y_test  : {:.1%}".format(y_test.mean()))
