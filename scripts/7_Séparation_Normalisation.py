import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Chargement et nettoyage préalable
df = pd.read_csv('data/cardio_data.csv', sep=';')
df = df.drop(columns=['id'])
df = df.drop_duplicates()
df = df[
    (df['ap_hi'] >= 50) & (df['ap_hi'] <= 250) &
    (df['ap_lo'] >= 30) & (df['ap_lo'] <= 200) &
    (df['ap_hi'] >= df['ap_lo'])
]
df['age_years'] = (df['age'] / 365.25).astype(int)
df = df.drop(columns=['age'])

# 2. Séparation des features (X) et de la cible (y)
X = df.drop(columns=['cardio'])
y = df['cardio']

# 3. Séparation en données d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalisation des données avec StandardScaler
# IMPORTANT : On ajuste (fit) le scaler uniquement sur X_train pour éviter le "data leakage" (fuite de données)
# et on l'applique ensuite sur X_train et X_test.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dimensions de X_train normalisé :", X_train_scaled.shape)
print("Dimensions de X_test normalisé :", X_test_scaled.shape)