# Sprint 3 - S3-T5 : Synthétiser les métriques dans un tableau
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

resultats = []
for nom, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    resultats.append({
        "Modèle": nom,
        "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Précision (%)": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall (%)": round(recall_score(y_test, y_pred) * 100, 2),
        "F1 (%)": round(f1_score(y_test, y_pred) * 100, 2),
        "ROC-AUC (%)": round(roc_auc_score(y_test, y_prob) * 100, 2),
    })

# Tableau trié par Recall (métrique prioritaire du projet)
tableau = pd.DataFrame(resultats).sort_values("Recall (%)", ascending=False)

print("=" * 80)
print("TABLEAU DE SYNTHÈSE DES MÉTRIQUES (jeu de test)")
print("=" * 80)
print(tableau.to_string(index=False))
print(f"\nMeilleur Recall : {tableau.iloc[0]['Modèle']} ({tableau.iloc[0]['Recall (%)']} %)")

tableau.to_csv(BASE / 'resultats' / 'tableau_metriques.csv', index=False, encoding='utf-8-sig')
print("\nTableau sauvegardé : resultats/tableau_metriques.csv")
