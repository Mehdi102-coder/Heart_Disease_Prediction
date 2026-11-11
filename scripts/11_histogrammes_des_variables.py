import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger le dataset nettoyé
df = pd.read_csv('cardio_data_processed.csv')

# 2. Configurer le style des graphiques
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 10))

# 3. Définir les variables numériques à tracer
numerical_cols = ['age_years', 'weight', 'ap_hi', 'ap_lo', 'height']

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, bins=30, color='teal')
    plt.title(f'Distribution de {col}', fontsize=12, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Effectif')

plt.tight_layout()

# 4. Sauvegarder l'image au lieu d'utiliser plt.show()
plt.savefig('distributions_variables.png', dpi=300)
plt.close()

print("Graphique sauvegardé avec succès sous 'distributions_variables.png' !")