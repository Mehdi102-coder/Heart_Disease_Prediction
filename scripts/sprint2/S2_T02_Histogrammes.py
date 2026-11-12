# Sprint 2 - S2-T2 : Créer les histogrammes des variables
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

variables = [c for c in df.columns if c != 'cardio']

fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, col in zip(axes.ravel(), variables):
    peu_de_valeurs = df[col].nunique() <= 10     # variables catégorielles / binaires
    sns.histplot(df[col], bins=30, kde=not peu_de_valeurs, discrete=peu_de_valeurs,
                 color='#9f1239', ax=ax)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('')

for ax in axes.ravel()[len(variables):]:
    ax.axis('off')

plt.suptitle("Histogrammes des variables", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE / 'resultats' / 'histogrammes_variables.png', dpi=300)
plt.close()
print("Image sauvegardée : resultats/histogrammes_variables.png")
