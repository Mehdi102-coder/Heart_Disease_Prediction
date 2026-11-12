# Sprint 1 - S1-T10 : Normaliser les variables numériques
import pandas as pd
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

colonnes_num = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']

print("--- Avant normalisation (moyenne, écart-type) ---")
print(df_enc[colonnes_num].agg(['mean', 'std']).round(2))

# StandardScaler : (x - moyenne) / écart-type
scaler = StandardScaler()
df_norm = df_enc.copy()
df_norm[colonnes_num] = scaler.fit_transform(df_enc[colonnes_num])

print("\n--- Après normalisation (moyenne ~ 0, écart-type ~ 1) ---")
print(df_norm[colonnes_num].agg(['mean', 'std']).round(3))

# Remarque : dans le pipeline final (S1-T11), le scaler est ajusté uniquement
# sur le jeu d'entraînement pour éviter toute fuite de données.
