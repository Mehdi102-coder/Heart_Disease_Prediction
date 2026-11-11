import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger le dataset nettoyé
df = pd.read_csv('cardio_data_processed.csv')

# 2. Calculer les effectifs et les pourcentages de la variable cible
class_counts = df['cardio'].value_counts()
class_percentages = df['cardio'].value_counts(normalize=True) * 100

print("--- Répartition absolue ---")
print(class_counts)

print("\n--- Répartition en pourcentage ---")
print(class_percentages)

# 3. Configurer le style de visualisation
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Graphique 1 : Diagramme en barres (Effectifs)
sns.countplot(x='cardio', data=df, palette='Set2', ax=axes[0])
axes[0].set_title("Répartition des classes (Effectifs)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Classe Cardio (0: Sain, 1: Malade)")
axes[0].set_ylabel("Nombre de patients")

# Ajouter les valeurs exactes sur les barres
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', 
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Graphique 2 : Diagramme circulaire (Camembert en pourcentages)
labels = ['Sain (0)', 'Malade (1)']
colors = sns.color_palette('Set2', len(labels))
axes[1].pie(class_counts, labels=labels, autopct='%1.2f%%', startangle=90, colors=colors, 
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title("Équilibre des classes (%)", fontsize=12, fontweight='bold')

plt.tight_layout()

# 4. Sauvegarder le graphique
plt.savefig('equilibre_classes.png', dpi=300)
plt.close()

print("\nGraphique de l'équilibre des classes sauvegardé sous 'equilibre_classes.png' avec succès !")