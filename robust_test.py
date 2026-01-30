import joblib
import numpy as np


        # 1. Load the saved .pkl files
model = joblib.load("knn_sport_model.pkl")
scaler = joblib.load("sport_scaler.pkl")
sport_names = joblib.load("sport_names.pkl")
        
        # 2. Get the original data from the model

X_original = model._fit_X
        
        # 3. Choose a test subject (e.g., the first sport in the list)
test_idx = 0 
original_dist, original_indices = model.kneighbors(X_original[test_idx].reshape(1, -1), n_neighbors=4)
original_neighbors = set(original_indices[0])

        # 4. Create a "Noisy" version of the model
X_noisy = X_original + np.random.normal(0, 0.5, X_original.shape)
        
        # Create a temporary model to test against the noise
from sklearn.neighbors import NearestNeighbors
noisy_model = NearestNeighbors(n_neighbors=4, metric=model.metric)
noisy_model.fit(X_noisy)
        
        # 5. Get neighbors from the noisy data
noisy_dist, noisy_indices = noisy_model.kneighbors(X_noisy[test_idx].reshape(1, -1), n_neighbors=4)
noisy_neighbors = set(noisy_indices[0])

        # 6. Calculate Score
overlap = original_neighbors.intersection(noisy_neighbors)
score = len(overlap) / len(original_neighbors)
        
print(f"--- Robustness Report ---")
print(f"Similarity Score: {score * 100:.1f}%")
        
if score >= 0.75:
    True
else: False