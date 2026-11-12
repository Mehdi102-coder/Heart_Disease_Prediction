# Sprint 3 - S3-T2 : Générer les matrices de confusion
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (nom, model) in zip(axes, models.items()):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n--- Matrice de confusion : {nom} ---")
    print(cm)
    print(f"Vrais négatifs : {tn} | Faux positifs : {fp} | Faux négatifs : {fn} | Vrais positifs : {tp}")

    ConfusionMatrixDisplay(cm, display_labels=["Sain (0)", "Malade (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(nom, fontweight="bold")

plt.suptitle("Matrices de confusion des modèles (jeu de test)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(BASE / 'resultats' / 'matrices_confusion.png', dpi=300)
plt.close()
print("\nImage sauvegardée : resultats/matrices_confusion.png")
