import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# 1. Charger et préparer les données
df = pd.read_csv('cardio_data_processed.csv')
X = df.drop(columns=['cardio'])
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. Entraîner le Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)

# 3. Calculer l'importance des variables
feature_importance = pd.DataFrame({
    'Variable': X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Affichage simple sans tabulate
print(feature_importance.to_string(index=False))

# 4. Générer la visualisation graphique
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Variable', data=feature_importance, palette='viridis')
plt.title("Importance des variables (Feature Importance) - Gradient Boosting", fontsize=14, fontweight='bold')
plt.xlabel("Importance relative")
plt.ylabel("Variables")
plt.tight_layout()

plt.savefig('feature_importance_gb.png', dpi=300)
plt.close()
print("\nGraphique sauvegardé sous 'feature_importance_gb.png' avec succès !")