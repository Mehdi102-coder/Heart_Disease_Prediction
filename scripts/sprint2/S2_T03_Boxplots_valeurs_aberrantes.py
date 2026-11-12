# Sprint 2 - S2-T3 : Créer les boxplots pour les valeurs aberrantes
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

colonnes_num = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for ax, col in zip(axes, colonnes_num):
    sns.boxplot(y=df[col], color='#fecdd3', ax=ax)
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('')
plt.suptitle("Boxplots : détection des valeurs aberrantes", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE / 'resultats' / 'boxplots_valeurs_aberrantes.png', dpi=300)
plt.close()

# Comptage des valeurs aberrantes avec la règle de l'IQR
print("--- Valeurs aberrantes (règle de l'IQR) ---")
for col in colonnes_num:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    bas, haut = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((df[col] < bas) | (df[col] > haut)).sum()
    print(f"{col:<10} bornes [{bas:7.1f} ; {haut:7.1f}] -> {n:5d} valeurs aberrantes ({n / len(df):.1%})")

print("\nImage sauvegardée : resultats/boxplots_valeurs_aberrantes.png")
