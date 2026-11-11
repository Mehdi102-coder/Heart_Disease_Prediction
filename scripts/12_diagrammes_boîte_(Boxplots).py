import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger le dataset nettoyé
df = pd.read_csv('cardio_data_processed.csv')

# 2. Configurer le style des graphiques
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 8))

# 3. Liste des variables numériques à analyser
numerical_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']

# 4. Tracer un boxplot pour chaque variable
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col], color='orange')
    plt.title(f'Boxplot de {col}', fontsize=12, fontweight='bold')
    plt.ylabel(col)

plt.tight_layout()

# 5. Sauvegarder l'image
plt.savefig('boxplots_valeurs_aberrantes.png', dpi=300)
plt.close()

print("Boxplots générés et sauvegardés sous 'boxplots_valeurs_aberrantes.png' avec succès !")