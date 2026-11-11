# S3-T5 : Synthétiser les métriques dans un tableau (FATIMAZAHRA)
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

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

# 4. Calcul des métriques pour chaque modèle
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results.append({
        "Modèle": name,
        "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Précision (%)": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall (%)": round(recall_score(y_test, y_pred) * 100, 2),
        "F1 (%)": round(f1_score(y_test, y_pred) * 100, 2),
        "ROC-AUC (%)": round(roc_auc_score(y_test, y_prob) * 100, 2),
    })

# 5. Tableau trié par Recall (métrique prioritaire du projet)
res_df = pd.DataFrame(results).sort_values("Recall (%)", ascending=False)

print("=" * 80)
print("TABLEAU DE SYNTHÈSE DES MÉTRIQUES (jeu de test)")
print("=" * 80)
print(res_df.to_string(index=False))

best = res_df.iloc[0]
print(f"\nMeilleur Recall : {best['Modèle']} ({best['Recall (%)']} %)")

# 6. Sauvegarde du tableau
res_df.to_csv("resultats/tableau_metriques.csv", index=False, encoding="utf-8-sig")
print("\nTableau sauvegardé sous 'resultats/tableau_metriques.csv'")
