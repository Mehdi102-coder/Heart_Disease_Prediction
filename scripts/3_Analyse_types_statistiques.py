import pandas as pd

# 1. Chargement du dataset
df = pd.read_csv('data/cardio_data.csv', sep=';')

# 2. Vérification des types de données
print("--- Types de données ---")
print(df.dtypes)

# 3. Statistiques descriptives pour les variables continues et discrètes
print("\n--- Statistiques descriptives ---")
print(df.describe())