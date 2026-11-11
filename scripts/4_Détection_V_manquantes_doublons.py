import pandas as pd

# 1. Chargement du dataset
df = pd.read_csv('data/cardio_data.csv', sep=';')

# 2. Vérification des valeurs manquantes par colonne
print("--- Valeurs manquantes par colonne ---")
missing_values = df.isnull().sum()
print(missing_values)
print(f"\nTotal des valeurs manquantes dans tout le dataset : {missing_values.sum()}")

# 3. Vérification des doublons
duplicates_count = df.duplicated().sum()
print(f"\nNombre de lignes en double (exactes) : {duplicates_count}")