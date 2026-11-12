# Sprint 2 - S2-T1 : Analyser la distribution des variables numériques
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

# Chargement du dataset préparé (produit par S1_T13_Exporter_dataset.py)
csv_prep = BASE / 'data' / 'cardio_data_processed.csv'
if not csv_prep.exists():
    csv_prep = BASE / 'cardio_data_processed.csv'
if not csv_prep.exists():
    raise SystemExit("Fichier introuvable : lance d'abord scripts/sprint1/S1_T13_Exporter_dataset.py")
df = pd.read_csv(csv_prep)

colonnes_num = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']

stats = df[colonnes_num].describe().T
stats['skewness'] = df[colonnes_num].skew()      # asymétrie
stats['kurtosis'] = df[colonnes_num].kurt()      # aplatissement
stats = stats.round(2)

print("--- Distribution des variables numériques ---")
print(stats)

print("\n--- Lecture de l'asymétrie ---")
for col in colonnes_num:
    s = df[col].skew()
    if abs(s) < 0.5:
        lecture = "distribution à peu près symétrique"
    elif s > 0:
        lecture = "étalée vers les grandes valeurs (queue à droite)"
    else:
        lecture = "étalée vers les petites valeurs (queue à gauche)"
    print(f"{col:<10} skewness = {s:6.2f} : {lecture}")

stats.to_csv(BASE / 'resultats' / 'distribution_variables_numeriques.csv', encoding='utf-8-sig')
print("\nTableau sauvegardé : resultats/distribution_variables_numeriques.csv")
