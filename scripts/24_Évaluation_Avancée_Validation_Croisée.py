import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

# 1. Charger les données
df = pd.read_csv('cardio_data_processed.csv')
X = df.drop(columns=['cardio'])
y = df['cardio']

# 2. Division Train / Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Dictionnaire des modèles évalués
models = {
    "Régression Logistique": LogisticRegression(random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# 5. Configuration de la Validation Croisée (5-Fold Stratifié)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    # Entraînement
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] # Pour le ROC-AUC
    
    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Validation croisée
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_acc_mean = cv_scores.mean()
    
    results.append({
        "Modèle": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Précision (%)": round(prec * 100, 2),
        "Rappel (%)": round(rec * 100, 2),
        "Score F1 (%)": round(f1 * 100, 2),
        "ROC-AUC (%)": round(roc_auc * 100, 2),
        "CV Accuracy (%)": round(cv_acc_mean * 100, 2)
    })
    
    print(f"\n--- Matrice de Confusion : {name} ---")
    print(confusion_matrix(y_test, y_pred))

# 6. Affichage du tableau récapitulatif
res_df = pd.DataFrame(results)
print("\n" + "="*70)
print("TABLEAU SYNTHÉTIQUE DES ÉVALUATIONS DE FIABILITÉ")
print("="*70)
print(res_df.to_string(index=False))

# 7. Génération de la Visualisation Graphique
plt.figure(figsize=(12, 6))
df_melted = res_df.melt(id_vars="Modèle", value_vars=["Accuracy (%)", "ROC-AUC (%)", "Score F1 (%)"], 
                        var_name="Métrique", value_name="Pourcentage (%)")

sns.barplot(x="Modèle", y="Pourcentage (%)", hue="Métrique", data=df_melted, palette="crest")
plt.title("Comparaison Visuelle des Performances Globales des Modèles", fontsize=14, fontweight='bold')
plt.ylim(60, 85)
plt.ylabel("Score (%)")
plt.xlabel("Modèles de Machine Learning")
plt.legend(title="Indicateurs clés", loc="upper right")
plt.tight_layout()

# Sauvegarde de l'image
plt.savefig("comparaison_modeles_performances.png", dpi=300)
plt.close()
print("\nGraphique des performances sauvegardé sous 'comparaison_modeles_performances.png' avec succès !")