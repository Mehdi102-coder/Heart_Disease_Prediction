import pandas as pd

# Chargement du dataset
df = pd.read_csv('data/cardio_data.csv', sep=';')

# Afficher un résumé statistique de toutes les variables
print(df.describe())