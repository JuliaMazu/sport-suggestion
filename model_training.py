import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn

# 1. Setup MLflow Experiment
mlflow.set_experiment("Sport_Recommendation_Engine")

with mlflow.start_run():
    # Load Data
    df = pd.read_csv('sport_data.csv')
    skill_columns = ['Endurance', 'Strength', 'Power', 'Speed', 'Agility',
           'Flexibility', 'Nerve', 'Durability', 'Hand-eye coordination',
           'Analytical Aptitude', 'Total']

    X = df[skill_columns]

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model Parameters
    n_neighbors = 4
    metric = 'euclidean'

    # Train Model
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    model.fit(X_scaled)

    # 2. Log Parameters and Metrics
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("distance_metric", metric)
    mlflow.log_metric("row_count", len(df))

    # 3. Save Local Files
    names = df['SPORT'].values
    joblib.dump(model, 'knn_sport_model.pkl')
    joblib.dump(scaler, 'sport_scaler.pkl')
    joblib.dump(names, 'sport_names.pkl')

    # 4. Log Artifacts to MLflow (The .pkl files)
    mlflow.log_artifact('knn_sport_model.pkl')
    mlflow.log_artifact('sport_scaler.pkl')
    mlflow.log_artifact('sport_names.pkl')
    
    # Optional: Log the model directly in MLflow format
    mlflow.sklearn.log_model(model, "model")

    print("Model, Scaler, and MLflow run saved successfully!")