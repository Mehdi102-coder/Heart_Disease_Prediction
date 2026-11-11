""" import pandas as pd

# (Reprise des étapes précédentes : chargement, nettoyage et feature engineering)
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

# 1. Analyse des variables catégorielles / ordinales existantes
# - gender : 1 ou 2 (binaire)
# - cholesterol : 1, 2, 3 (ordinal : respecte un ordre croissant de gravité)
# - gluc : 1, 2, 3 (ordinal)
# - smoke, alco, active : 0 ou 1 (binaires)

# 2. Encodage One-Hot (si on souhaite transformer 'gender' en colonnes binaires indépendantes)
# Note : Pour 'gender' (2 modalités) ou des variables binaires, ce n'est pas strictement obligatoire,
# mais c'est une bonne pratique pour certains modèles si on veut éviter toute notion de hiérarchie numérique.
df_encoded = pd.get_dummies(df, columns=['gender'], prefix='gender', drop_first=True)

print("--- Aperçu des colonnes après encodage ---")
print(df_encoded.head())
print(df_encoded.columns)"""