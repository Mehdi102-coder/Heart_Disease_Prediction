import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger le dataset nettoyé
df = pd.read_csv('cardio_data_processed.csv')

# 2. Définir les variables numériques à analyser
numerical_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']

# 3. Afficher les statistiques descriptives (moyenne, écart-type, min, médiane, max)
print("--- Statistiques descriptives des variables numériques ---")
stats = df[numerical_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
stats.rename(columns={'50%': 'median'}, inplace=True)
print(stats)

# 4. Générer et sauvegarder les graphiques de distribution (Histogrammes + Courbes de densité KDE)
plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, bins=30, color='royalblue')
    plt.title(f'Distribution de {col}', fontsize=12, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Fréquence')

plt.tight_layout()
plt.savefig('distributions_numeriques.png', dpi=300)
plt.show()
print("\nGraphique de distribution sauvegardé sous 'distributions_numeriques.png'.")