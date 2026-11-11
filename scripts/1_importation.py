import pandas as pd

# 1. Chargement du dataset (le séparateur est un point-virgule ';')
df = pd.read_csv('data/cardio_data.csv', sep=';')

# 2. Afficher les dimensions (nombre de lignes et de colonnes)
print(f"Dimensions du dataset : {df.shape}")

# 3. Afficher les premières lignes
print("\n--- Premières lignes ---")
print(df.head())

# 4. Afficher les types de données et vérifier s'il y a des valeurs nulles
print("\n--- Informations générales ---")
print(df.info())

# 5. Vérifier la répartition de la variable cible (cardio)
print("\n--- Répartition de la cible (cardio) ---")
print(df['cardio'].value_counts())