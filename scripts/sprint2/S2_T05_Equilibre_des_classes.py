# Sprint 2 - S2-T5 : Analyser l'équilibre des classes (malades / non-malades)
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

comptes = df['cardio'].value_counts().sort_index()
pourcentages = df['cardio'].value_counts(normalize=True).sort_index() * 100

print("--- Équilibre des classes ---")
print(f"Non-malades (0) : {comptes[0]} ({pourcentages[0]:.1f} %)")
print(f"Malades     (1) : {comptes[1]} ({pourcentages[1]:.1f} %)")
ecart = abs(pourcentages[0] - pourcentages[1])
print(f"Écart entre les classes : {ecart:.1f} points")
print("Conclusion :", "classes équilibrées (pas besoin de rééchantillonnage)" if ecart < 10
      else "classes déséquilibrées (envisager un rééquilibrage)")

plt.figure(figsize=(6, 5))
ax = sns.countplot(x='cardio', data=df, hue='cardio', palette=['#93c5fd', '#9f1239'], legend=False)
for i, v in enumerate(comptes):
    ax.text(i, v, f"{v}\n({pourcentages[i]:.1f} %)", ha='center', va='bottom')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Non-malade (0)', 'Malade (1)'])
ax.set_xlabel('')
ax.set_ylabel("Nombre de patients")
plt.title("Équilibre des classes", fontweight='bold')
plt.ylim(0, comptes.max() * 1.15)
plt.tight_layout()
plt.savefig(BASE / 'resultats' / 'equilibre_classes.png', dpi=300)
plt.close()
print("\nImage sauvegardée : resultats/equilibre_classes.png")
