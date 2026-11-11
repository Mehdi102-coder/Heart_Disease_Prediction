# S3-T3 : Tracer les courbes ROC et calculer l'AUC (FATIMAZAHRA)
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score

os.makedirs('resultats', exist_ok=True)

# 1. Charger le dataset préparé
df = pd.read_csv('cardio_data_processed.csv')
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

# 4. Courbe ROC de chaque modèle (on utilise les probabilités de la classe 1)
plt.figure(figsize=(8, 7))
print("--- AUC par modèle ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name} : AUC = {auc:.4f}")
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")

# 5. Ligne de référence : classifieur aléatoire (AUC = 0.5)
plt.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0.500)")
plt.xlabel("Taux de faux positifs (1 - Spécificité)")
plt.ylabel("Taux de vrais positifs (Rappel)")
plt.title("Courbes ROC des modèles (jeu de test)", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("resultats/courbes_roc_auc.png", dpi=300)
plt.close()
print("\nImage sauvegardée sous 'resultats/courbes_roc_auc.png'")
