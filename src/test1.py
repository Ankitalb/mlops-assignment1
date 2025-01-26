import mlflow
import mlflow.sklearn
from flask import Flask, jsonify, request


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

model_uri = "runs:/4b413a39e1b740e0a41fac0fdd14c68e/random_forest_model2"
model = mlflow.sklearn.load_model(model_uri)

data = pd.read_csv('C:/Users/ankita/Documents/mlops/Assignement1/data/dataset.csv')
df = pd.DataFrame(data)
print(df.columns.values.tolist())

input_data = ['7500', '4', '2', '2', 'yes', 'no', 'yes', 'no', 'yes', '3', 'yes',
 'furnished']
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

# Drop the original 'furnishingstatus' column if needed
df = df.drop(columns=['furnishingstatus'])
print(df)

prediction = model.predict(df)
print(prediction)
        # Make predictions
        
        # Return the prediction as JSON
