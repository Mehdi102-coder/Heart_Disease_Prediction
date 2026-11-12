# Sprint 1 - S1-T12 : Vérifier et valider le dataset préparé
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


df_final = encoder(nettoyer(df))

# Les doublons se vérifient avec l'âge en jours (deux patients du même âge en années
# ne sont pas des doublons : leur âge exact en jours diffère).
brut_conserve = df.drop(columns=['id']).loc[df_final.index]

controles = {
    "Aucune valeur manquante": df_final.isnull().sum().sum() == 0,
    "Aucun doublon exact (âge en jours inclus)": brut_conserve.duplicated().sum() == 0,
    "Toutes les colonnes sont numériques": all(str(t) != 'object' for t in df_final.dtypes),
    "Cible binaire (0/1)": set(df_final['cardio'].unique()) == {0, 1},
    "gender encodé en 0/1": set(df_final['gender'].unique()) == {0, 1},
    "ap_hi >= ap_lo partout": (df_final['ap_hi'] >= df_final['ap_lo']).all(),
    "ap_hi dans [50, 250]": df_final['ap_hi'].between(50, 250).all(),
    "ap_lo dans [30, 200]": df_final['ap_lo'].between(30, 200).all(),
    "Colonne 'age' remplacée par 'age_years'": 'age' not in df_final.columns and 'age_years' in df_final.columns,
    "Colonne 'id' supprimée": 'id' not in df_final.columns,
}

print("--- Contrôles de validation ---")
for nom, ok in controles.items():
    print(f"[{'OK' if ok else 'ERREUR'}] {nom}")

print(f"\nDimensions finales : {df_final.shape}")
print("Répartition de la cible :")
print(df_final['cardio'].value_counts(normalize=True).round(3))

if all(controles.values()):
    print("\nDataset VALIDE : prêt à être exporté (S1-T13).")
else:
    print("\nDataset NON valide : corriger les erreurs ci-dessus.")
