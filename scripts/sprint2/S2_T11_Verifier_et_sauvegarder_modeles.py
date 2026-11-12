# Sprint 2 - S2-T11 : Vérifier les résultats et sauvegarder les modèles
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

dossier_modeles = BASE / 'models'
dossier_modeles.mkdir(exist_ok=True)

for nom, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Vérifications de cohérence
    assert len(y_pred) == len(y_test), "Nombre de prédictions incorrect"
    assert set(np.unique(y_pred)) <= {0, 1}, "Prédictions non binaires"
    assert acc > 0.5, "Le modèle fait pire que le hasard"
    proba = model.predict_proba(X_test_scaled)
    assert np.allclose(proba.sum(axis=1), 1), "Les probabilités ne somment pas à 1"

    # Sauvegarde (compress=3 : le fichier Random Forest reste léger pour GitHub)
    fichier = dossier_modeles / (nom.lower().replace(' ', '_').replace('(', '').replace(')', '')
                                 .replace('=', '').replace('é', 'e').replace('è', 'e') + '_model.pkl')
    joblib.dump(model, fichier, compress=3)

    # Rechargement : les prédictions doivent être identiques
    recharge = joblib.load(fichier)
    assert (recharge.predict(X_test_scaled) == y_pred).all(), "Modèle rechargé différent"

    print(f"[OK] {nom:<22} accuracy = {acc:.4f} | {fichier.name} ({fichier.stat().st_size / 1e6:.1f} Mo)")

joblib.dump(scaler, dossier_modeles / 'scaler.pkl')
print("[OK] Scaler sauvegardé : models/scaler.pkl")
print("\nTous les modèles ont été vérifiés et sauvegardés dans le dossier 'models/'.")
