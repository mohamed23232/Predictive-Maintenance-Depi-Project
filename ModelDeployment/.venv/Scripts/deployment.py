from flask import Flask, render_template
from flask import request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model_path = r"voting_classifier_model.pkl"
scalar = joblib.load("scaler.pkl")
model = joblib.load(model_path)

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the data from the request 
    air_temperature = float(request.form['air_temperature'])
    rotational_speed = float(request.form['rotational_speed'])
    torque = float(request.form['torque'])
    tool_wear = float(request.form['tool_wear'])
    process_temperature = float(request.form['process_temperature'])
    type_value = int(request.form['type'])

    features = ['torque', 'tool_wear', 'air_temperature', 'rotational_speed', 'process_temperature', 'type_enc']
    # Store the inputs in a list
    input_data = [
        torque,
        tool_wear,
        air_temperature,
        rotational_speed,
        process_temperature,
        type_value
    ]

    df_data = pd.DataFrame([input_data], columns=features)

    # convert the input data into a numpy array
    data = scalar.transform(df_data)
    # make a prediction
    prediction = model.predict(data)
    # return the prediction
    prediction_text = ""
    if prediction[0] == 1:
        prediction_text = "The machine will FAIL"
    else:
        prediction_text = "The machine will NOT FAIL"

    return render_template('index.html', prediction=f'{prediction_text}')


if __name__ == '__main__':
    app.run(debug=True)