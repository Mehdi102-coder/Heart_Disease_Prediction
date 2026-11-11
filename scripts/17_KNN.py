import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Charger le dataset préparé
df = pd.read_csv('cardio_data_processed.csv')

# 2. Séparer les caractéristiques (X) de la cible (y)
X = df.drop(columns=['cardio'])
y = df['cardio']

# 3. Diviser les données en ensembles d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normaliser les données (crucial pour le calcul des distances dans KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Instancier et entraîner le modèle KNN (par exemple avec k = 5 voisins)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# 6. Faire des prédictions sur l'ensemble de test
y_pred = knn_model.predict(X_test_scaled)

# 7. Évaluer les performances
print(f"Précision (Accuracy) : {accuracy_score(y_test, y_pred):.4f}\n")
print("Matrice de Confusion :")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de Classification :")
print(classification_report(y_test, y_pred))