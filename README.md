# Prédiction des Maladies Cardiaques (Scrum)

Un script Python par tâche, rangé par sprint. Chaque script est autonome :
il retrouve tout seul la racine du projet (le dossier qui contient `data/`),
donc il se lance depuis n'importe quel dossier, sur n'importe quel ordinateur.

## Installation (une seule fois)

    python -m pip install -r requirements.txt

## Organisation

    data/                  cardio_data.csv (brut) et cardio_data_processed.csv (préparé)
    scripts/sprint1/       S1-T4 à S1-T13  : chargement, nettoyage, préparation
    scripts/sprint2/       S2-T1 à S2-T11  : analyse exploratoire, 3 modèles de base
    scripts/sprint3/       S3-T1 à S3-T9   : évaluation et comparatif de modèles
    resultats/             images et tableaux produits (créé automatiquement)
    models/                modèles sauvegardés (créé par S2-T11, non versionné)

## Ordre d'exécution

- Sprint 1 : les scripts lisent le fichier brut `data/cardio_data.csv`.
  S1-T13 crée `data/cardio_data_processed.csv`.
- Sprint 2 et 3 : les scripts lisent `data/cardio_data_processed.csv`
  (déjà fourni ; à régénérer avec S1-T13 si besoin).

## Durées indicatives

La plupart des scripts prennent moins de 15 secondes. S3-T4 (validation croisée)
prend environ 30 secondes. S3-T6 (SVM) et S3-T9 (comparaison) prennent 2 minutes environ.

## Remarque

S3-T8 et S3-T9 utilisent XGBoost : `python -m pip install xgboost`.
Sans XGBoost, S3-T9 compare les autres modèles et ignore XGBoost.
