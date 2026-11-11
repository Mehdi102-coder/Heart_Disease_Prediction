import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os

# 1. Créer un dossier 'models' s'il n'existe pas
os.makedirs('models', exist_ok=True)

# 2. Charger le dataset nettoyé
df = pd.read_csv('cardio_data_processed.csv')

# 3. Séparer les features et la cible
X = df.drop(columns=['cardio'])
y = df['cardio']

# 4. Diviser les données (Train / Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normaliser les données (Crucial pour la Régression Logistique et le KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Sauvegarder le scaler pour l'utiliser sur de futures nouvelles données
joblib.dump(scaler, 'models/scaler.pkl')
print("Scaler sauvegardé sous 'models/scaler.pkl'")

# 6. Dictionnaire contenant tous nos modèles
models = {
    "logistic_regression": LogisticRegression(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42)
}

# 7. Entraîner et sauvegarder chaque modèle
for name, model in models.items():
    print(f"Entraînement de {name}...")
    model.fit(X_train_scaled, y_train)
    
    filename = f'models/{name}_model.pkl'
    joblib.dump(model, filename)
    print(f"-> Modèle sauvegardé avec succès sous '{filename}'")

print("\nTous les modèles et le scaler ont été sauvegardés dans le dossier 'models/' avec succès !")