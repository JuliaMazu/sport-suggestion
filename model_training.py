import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('sport_data.csv')
skill_columns = ['Endurance', 'Strength', 'Power', 'Speed', 'Agility',
       'Flexibility', 'Nerve', 'Durability', 'Hand-eye coordination',
       'Analytical Aptitude', 'Total']

X = df[skill_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = NearestNeighbors(n_neighbors=4, metric='euclidean')
model.fit(X_scaled)

import joblib
names = df['SPORT'].values

joblib.dump(model, 'knn_sport_model.pkl')
joblib.dump(scaler, 'sport_scaler.pkl')
joblib.dump(names, 'sport_names.pkl')

print("Model and Scaler saved successfully!")