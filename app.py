from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load assets
model = joblib.load('knn_sport_model.pkl')
scaler = joblib.load('sport_scaler.pkl')
sport_names = joblib.load('sport_names.pkl')

@app.route('/suggest/<sport_name>')
def suggest_by_name(sport_name):
    try:
        # Find any sport name that CONTAINS the input string
        mask = [sport_name.lower() in str(n).lower() for n in sport_names]
        idx = np.where(mask)[0][0] 
        
        # ... rest of your code ...
        distances, indices = model.kneighbors(model._fit_X[idx].reshape(1, -1), n_neighbors=4)
        suggestions = [sport_names[i] for i in indices[0][1:]]
        
        return jsonify({
            "found_match": sport_names[idx], 
            "suggestions": suggestions
        })
    except:
        return jsonify({"error": f"No sport containing '{sport_name}' found"}), 404

@app.route('/predict', methods=['POST'])
def predict_by_skills():
    # Expecting a JSON list of 10 numbers: {"skills": [8, 5, 4, ...]}
    data = request.get_json()
    if not data or 'skills' not in data:
        return jsonify({"error": "Please provide a 'skills' list"}), 400

    scaled_skills = scaler.transform([data['skills']])
    distances, indices = model.kneighbors(scaled_skills, n_neighbors=3)
    
    suggestions = [sport_names[i] for i in indices[0]]
    return jsonify({"suggestions": suggestions})

if __name__ == '__main__':
    app.run(debug=True)