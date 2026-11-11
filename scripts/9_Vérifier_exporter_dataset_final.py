import pandas as pd

# 1. Chargement du dataset d'origine
df = pd.read_csv('data/cardio_data.csv', sep=';')

# 2. Suppression de l'ID et des doublons
df = df.drop(columns=['id']).drop_duplicates()

# 3. Filtrage des valeurs aberrantes de pression artérielle
df = df[
    (df['ap_hi'] >= 50) & (df['ap_hi'] <= 250) &
    (df['ap_lo'] >= 30) & (df['ap_lo'] <= 200) &
    (df['ap_hi'] >= df['ap_lo'])
]

# 4. Feature Engineering : Conversion de l'âge en années
df['age_years'] = (df['age'] / 365.25).astype(int)
df = df.drop(columns=['age'])

# 5. Exportation du dataset préparé vers un nouveau fichier CSV
output_filename = 'cardio_data_processed.csv'
df.to_csv(output_filename, index=False)

print(f"Dataset nettoyé et exporté avec succès sous : '{output_filename}'")
print(f"Nombre total de lignes finales : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")