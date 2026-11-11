import pandas as pd

# 1. Chargement du dataset
df = pd.read_csv('data/cardio_data.csv', sep=';')

# 2. Suppression des doublons (hors ID)
df = df.drop(columns=['id'])
df = df.drop_duplicates()

# 3. Filtrer les valeurs aberrantes pour la pression artérielle (ap_hi et ap_lo)
# Une pression systolique (ap_hi) normale/plausible se situe généralement entre 50 et 250 mmHg.
# Une pression diastolique (ap_lo) normale/plausible se situe généralement entre 30 et 200 mmHg.
# On élimine également les cas où la systolique est inférieure à la diastolique (physiologiquement incohérent).
df_cleaned = df[
    (df['ap_hi'] >= 50) & (df['ap_hi'] <= 250) &
    (df['ap_lo'] >= 30) & (df['ap_lo'] <= 200) &
    (df['ap_hi'] >= df['ap_lo'])
]

# 4. Feature Engineering : Conversion de l'âge en années (plus interprétable)
df_cleaned['age_years'] = (df_cleaned['age'] / 365.25).astype(int)
df_cleaned = df_cleaned.drop(columns=['age']) # On supprime l'ancienne colonne en jours

print(f"Dimensions après nettoyage : {df_cleaned.shape}")
print(df_cleaned.head())