# Sprint 2 - S2-T4 : Créer une matrice de corrélation (Seaborn)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

corr = df.corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title("Matrice de corrélation", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE / 'resultats' / 'matrice_correlation.png', dpi=300)
plt.close()

print("--- Corrélation de chaque variable avec la cible (cardio) ---")
corr_cible = corr['cardio'].drop('cardio').sort_values(key=abs, ascending=False)
print(corr_cible.round(3))
print("\nImage sauvegardée : resultats/matrice_correlation.png")
