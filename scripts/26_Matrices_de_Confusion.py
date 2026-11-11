# S3-T2 : Générer les matrices de confusion (FATIMAZAHRA)
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 
# 0. Chemins relatifs : BASE = dossier du projet (parent du dossier "scripts")
BASE = Path(__file__).resolve().parent.parent
os.makedirs(BASE / 'resultats', exist_ok=True)
 
# 1. Charger le dataset préparé (dans data/, sinon à la racine du projet)
csv_path = BASE / 'data' / 'cardio_data_processed.csv'
if not csv_path.exists():
    csv_path = BASE / 'cardio_data_processed.csv'
df = pd.read_csv(csv_path)
X = df.drop(columns=['cardio'])
y = df['cardio']
 
# 2. Même découpage et même normalisation que les autres scripts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# 3. Modèles évalués
models = {
    "Régression Logistique": LogisticRegression(random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
 
# 4. Entraîner, prédire et tracer une matrice par modèle
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (name, model) in zip(axes.ravel(), models.items()):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
 
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n--- Matrice de Confusion : {name} ---")
    print(cm)
    print(f"Vrais négatifs : {tn} | Faux positifs : {fp} | Faux négatifs : {fn} | Vrais positifs : {tp}")
 
    ConfusionMatrixDisplay(cm, display_labels=["Sain (0)", "Malade (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(name, fontweight="bold")
 
plt.suptitle("Matrices de confusion des modèles (jeu de test)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(BASE / 'resultats' / 'matrices_confusion.png', dpi=300)
plt.close()
print("\nImage sauvegardée sous 'resultats/matrices_confusion.png'")
