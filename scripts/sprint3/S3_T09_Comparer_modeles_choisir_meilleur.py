# Sprint 3 - S3-T9 : Comparer les modèles et choisir le meilleur
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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

# Les 6 modèles à comparer (le SVM peut prendre 1 à 2 minutes)
models = {
    "Régression Logistique": LogisticRegression(random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                                      eval_metric='logloss', random_state=42)
except ImportError:
    print("XGBoost n'est pas installé (pip install xgboost) : il est ignoré dans la comparaison.\n")

resultats = []
for nom, model in models.items():
    print(f"Évaluation : {nom} ...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # Score continu pour l'AUC : probabilités, ou distance à l'hyperplan pour le SVM
    if hasattr(model, 'predict_proba'):
        score = model.predict_proba(X_test_scaled)[:, 1]
    else:
        score = model.decision_function(X_test_scaled)

    resultats.append({
        "Modèle": nom,
        "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Précision (%)": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall (%)": round(recall_score(y_test, y_pred) * 100, 2),
        "F1 (%)": round(f1_score(y_test, y_pred) * 100, 2),
        "ROC-AUC (%)": round(roc_auc_score(y_test, score) * 100, 2),
    })

# Classement : le Recall est la métrique prioritaire du projet (ne pas manquer de malades),
# le F1 départage les ex aequo.
tableau = pd.DataFrame(resultats).sort_values(["Recall (%)", "F1 (%)"], ascending=False)

print("\n" + "=" * 85)
print("COMPARAISON DES MODÈLES (jeu de test)")
print("=" * 85)
print(tableau.to_string(index=False))

meilleur = tableau.iloc[0]
print(f"\nMEILLEUR MODÈLE (critère : Recall) : {meilleur['Modèle']}")
print(f"  Recall = {meilleur['Recall (%)']} % | F1 = {meilleur['F1 (%)']} % | "
      f"ROC-AUC = {meilleur['ROC-AUC (%)']} %")
print(f"Meilleure AUC : {tableau.sort_values('ROC-AUC (%)', ascending=False).iloc[0]['Modèle']}")

tableau.to_csv(BASE / 'resultats' / 'comparaison_modeles.csv', index=False, encoding='utf-8-sig')

# Graphique comparatif
long = tableau.melt(id_vars="Modèle", value_vars=["Recall (%)", "F1 (%)", "ROC-AUC (%)"],
                    var_name="Métrique", value_name="Score (%)")
plt.figure(figsize=(12, 6))
sns.barplot(x="Modèle", y="Score (%)", hue="Métrique", data=long, palette="rocket")
plt.title("Comparaison des modèles", fontsize=14, fontweight="bold")
plt.ylim(50, 90)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(BASE / 'resultats' / 'comparaison_modeles.png', dpi=300)
plt.close()
print("\nFichiers sauvegardés : resultats/comparaison_modeles.csv et comparaison_modeles.png")
