import joblib
import pandas as pd

# 1. Charger le modèle et le scaler
model = joblib.load('models/gradient_boosting_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# 2. Définir l'ordre exact des colonnes
colonnes = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# 3. Créer le DataFrame
nouveau_patient_df = pd.DataFrame([[
    50.0,   # age_years
    1,      # gender
    168,    # height
    62.0,   # weight
    120,    # ap_hi
    80,     # ap_lo
    1,      # cholesterol
    1,      # gluc
    0,      # smoke
    0,      # alco
    1       # active
]], columns=colonnes)

# 4. Convertir en tableau NumPy (.values) pour contourner la vérification des noms de colonnes
patient_scaled = scaler.transform(nouveau_patient_df.values)

# 5. Faire la prédiction
prediction = model.predict(patient_scaled)
proba = model.predict_proba(patient_scaled)

print(f"Prédiction : {'Malade (Risque)' if prediction[0] == 1 else 'Sain'}")
print(f"Probabilité : {proba[0][1]*100:.2f}% de risque cardiovasculaire")