🏅 Sport Similarity Engine (KNN)This project uses a K-Nearest Neighbors model to suggest sports based on a profile of 10 physical skills. 

1. Start with the download data file. It will download the data to the same working directory.

This should be changed to the S3 bucket



2. Check_data - returns true or false for the data quality and some prints about errors (if appliable)



3. Model training - 2 in 1. Prepares data and creates .plk files (models) which will be used
The same issue as with the data, models are saves to the working directory, this should be modified


4. robust test - returns true false, to simplify our life. 



5. app.py - how model is presented to the endpoint (I am not sure if you need it or not).

Endpoints
🟢 GET: Suggest by Name
Find sports similar to one already in our database.

URL: http://127.0.0.1:5000/suggest/<sport_name>

Example: http://127.0.0.1:5000/suggest/rugby

🔵 POST: Predict by Skills
Predict the best sport for a custom set of 10 skill ratings.

URL: http://127.0.0.1:5000/predict

Payload: A JSON object containing a list of 10 floats.

Example Request Script:

url = "http://127.0.0.1:5000/predict"
my_skills = {
    "skills": [8.5, 7.0, 4.0, 9.0, 6.5, 2.0, 8.0, 5.0, 9.0, 7.0]
}

response = requests.post(url, json=my_skills)

if response.status_code == 200:
    print("Suggestions:", response.json()['suggestions'])
else:
    print("Error:", response.text)