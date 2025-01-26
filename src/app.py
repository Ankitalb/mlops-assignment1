from flask import Flask, jsonify, request
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Create an instance of the Flask class
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask app! branch..."

import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import numpy as np

# Initialize Flask app
app = Flask(__name__)

model_uri = "runs:/4b413a39e1b740e0a41fac0fdd14c68e/random_forest_model2"
model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert the input data to a numpy array (make sure it's shaped correctly)
        input = np.array(data['input']).reshape(1, -1)
        input_data = input[0]
        print(input_data)
        column_names = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
        'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
        df = pd.DataFrame([input_data], columns=column_names)

        encoder = LabelEncoder()
        df['mainroad'] = encoder.fit_transform(df['mainroad'])
        df['guestroom'] = encoder.fit_transform(df['guestroom'])
        df['basement'] = encoder.fit_transform(df['basement'])
        df['hotwaterheating'] = encoder.fit_transform(df['hotwaterheating'])
        df['airconditioning'] = encoder.fit_transform(df['airconditioning'])
        df['prefarea'] = encoder.fit_transform(df['prefarea'])

        df['furnishingstatus_furnished'] = (df['furnishingstatus'] == 'furnished')
        df['furnishingstatus_semi-furnished'] = (df['furnishingstatus'] == 'semi-furnished')
        df['furnishingstatus_unfurnished'] = (df['furnishingstatus'] == 'unfurnished')

        df = df.drop(columns=['furnishingstatus'])
        print(df)

        prediction = model.predict(df)
        print(prediction)
    
        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
