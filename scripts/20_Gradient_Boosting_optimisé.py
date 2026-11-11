import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Charger le dataset préparé
df = pd.read_csv('cardio_data_processed.csv')

# 2. Séparer features et cible
X = df.drop(columns=['cardio'])
y = df['cardio']

# 3. Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Instanciation d'un modèle de Gradient Boosting avec hyperparamètres affinés
gb_tuned = GradientBoostingClassifier(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42
)
gb_tuned.fit(X_train_scaled, y_train)

# 6. Prédictions et évaluation
y_pred = gb_tuned.predict(X_test_scaled)

print(f"Précision (Accuracy) : {accuracy_score(y_test, y_pred):.4f}\n")
print("Matrice de Confusion :")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de Classification :")
print(classification_report(y_test, y_pred))