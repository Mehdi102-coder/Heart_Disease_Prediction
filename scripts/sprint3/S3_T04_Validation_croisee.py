# Sprint 3 - S3-T4 : Réaliser la validation croisée
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

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

# Séparation X / y, découpage 80/20 et normalisation (fit sur le train uniquement)
X = df.drop(columns=['cardio'])
y = df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Régression Logistique": LogisticRegression(random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Validation croisée stratifiée à 5 plis, sur le jeu d'entraînement
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metriques = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

resultats = []
for nom, model in models.items():
    print(f"Validation croisée : {nom} ...")
    scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=metriques)
    ligne = {"Modèle": nom}
    for m in metriques:
        s = scores[f'test_{m}']
        ligne[f"{m} (moyenne %)"] = round(s.mean() * 100, 2)
        ligne[f"{m} (écart-type)"] = round(s.std() * 100, 2)
    resultats.append(ligne)

tableau = pd.DataFrame(resultats)
print("\n" + "=" * 80)
print("RÉSULTATS DE LA VALIDATION CROISÉE (5 plis)")
print("=" * 80)
print(tableau.T.to_string(header=False))

tableau.to_csv(BASE / 'resultats' / 'validation_croisee.csv', index=False, encoding='utf-8-sig')
print("\nTableau sauvegardé : resultats/validation_croisee.csv")
print("Un petit écart-type = un modèle stable d'un pli à l'autre.")
