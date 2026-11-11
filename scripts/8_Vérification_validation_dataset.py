import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Chargement
df = pd.read_csv('data/cardio_data.csv', sep=';')

# 2. Nettoyage et doublons
df = df.drop(columns=['id']).drop_duplicates()

# 3. Filtrage des pressions artérielles aberrantes
df = df[
    (df['ap_hi'] >= 50) & (df['ap_hi'] <= 250) &
    (df['ap_lo'] >= 30) & (df['ap_lo'] <= 200) &
    (df['ap_hi'] >= df['ap_lo'])
]

# 4. Feature Engineering sur l'âge
df['age_years'] = (df['age'] / 365.25).astype(int)
df = df.drop(columns=['age'])

# 5. Séparation Features / Target
X = df.drop(columns=['cardio'])
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=== DATASET VALIDÉ AVEC SUCCÈS ===")
print(f"Forme de X_train (scaled) : {X_train_scaled.shape}")
print(f"Forme de X_test (scaled)  : {X_test_scaled.shape}")
print(f"Valeurs nulles : {pd.isnull(X_train_scaled).sum()}")