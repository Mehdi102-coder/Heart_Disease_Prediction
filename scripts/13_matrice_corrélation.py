import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger le dataset nettoyé
df = pd.read_csv('cardio_data_processed.csv')

# 2. Calculer la matrice de corrélation
corr_matrix = df.corr()

# Afficher les corrélations triées par rapport à la cible 'cardio'
print("--- Corrélation avec la cible (cardio) ---")
print(corr_matrix['cardio'].sort_values(ascending=False))

# 3. Générer et sauvegarder la Heatmap de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Matrice de Corrélation des Variables', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('matrice_correlation.png', dpi=300)
plt.close()

print("\nHeatmap de corrélation sauvegardée sous 'matrice_correlation.png' !")