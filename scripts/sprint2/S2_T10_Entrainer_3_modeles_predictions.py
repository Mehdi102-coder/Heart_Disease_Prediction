# Sprint 2 - S2-T10 : Entraîner les 3 modèles et réaliser les prédictions
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

# Entraînement de chaque modèle et prédictions sur le jeu de test
predictions = pd.DataFrame({'y_reel': y_test.values})
for nom, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions[nom] = model.predict(X_test_scaled)
    print(f"{nom:<22} accuracy = {accuracy_score(y_test, predictions[nom]):.4f}")

print("\n--- Aperçu des prédictions (10 premiers patients) ---")
print(predictions.head(10))

predictions.to_csv(BASE / 'resultats' / 'predictions_modeles.csv', index=False)
print("\nPrédictions sauvegardées : resultats/predictions_modeles.csv")
