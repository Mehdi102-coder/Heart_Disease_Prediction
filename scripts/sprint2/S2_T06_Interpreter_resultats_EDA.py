# Sprint 2 - S2-T6 : Interpréter les résultats de l'EDA
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

lignes = []

# 1. Équilibre des classes
p = df['cardio'].mean() * 100
lignes.append(f"1. Classes : {p:.1f} % de malades et {100 - p:.1f} % de non-malades. "
              f"Le jeu de données est {'équilibré' if abs(50 - p) < 5 else 'déséquilibré'}, "
              "l'accuracy est donc une métrique exploitable.")

# 2. Corrélation avec la cible
corr = df.corr()['cardio'].drop('cardio').sort_values(key=abs, ascending=False)
top = ", ".join(f"{k} ({v:+.2f})" for k, v in corr.head(3).items())
lignes.append(f"2. Variables les plus liées à la maladie : {top}.")
lignes.append(f"   Variable la moins liée : {corr.abs().idxmin()} ({corr[corr.abs().idxmin()]:+.2f}).")

# 3. Corrélations entre variables explicatives
x = df.drop(columns=['cardio']).corr().abs()
paires = [(a, b, x.loc[a, b]) for i, a in enumerate(x.columns) for b in x.columns[i + 1:]]
a, b, v = max(paires, key=lambda t: t[2])
lignes.append(f"3. Corrélation la plus forte entre variables explicatives : {a} / {b} ({v:.2f}).")

# 4. Valeurs aberrantes (règle de l'IQR)
for col in ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    lignes.append(f"4. {col} : {n / len(df):.1%} de valeurs hors bornes IQR, asymétrie = {df[col].skew():+.2f}.")

# 5. Profil moyen selon la cible
moy = df.groupby('cardio')[['age_years', 'weight', 'ap_hi', 'ap_lo']].mean().round(1)
lignes.append("5. Moyennes selon la cible (0 = sain, 1 = malade) :\n" + moy.to_string())

texte = "INTERPRÉTATION DE L'ANALYSE EXPLORATOIRE (EDA)\n" + "=" * 50 + "\n" + "\n".join(lignes)
print(texte)

(BASE / 'resultats' / 'interpretation_EDA.txt').write_text(texte, encoding='utf-8')
print("\nTexte sauvegardé : resultats/interpretation_EDA.txt")
