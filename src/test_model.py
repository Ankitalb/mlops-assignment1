import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def test_model():
    MODEL_URI = "runs:/4b413a39e1b740e0a41fac0fdd14c68e/random_forest_model2"
    model = mlflow.sklearn.load_model(MODEL_URI)
    data = {"input": [7500, 4, 2, 2, "yes", "no", "yes", "no", "yes", 3, "yes",
                      "furnished"]}
    input = np.array(data['input']).reshape(1, -1)
    input_data = input[0]
    column_names = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
                    'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'parking', 'prefarea',
                    'furnishingstatus']
    df = pd.DataFrame([input_data], columns=column_names)

    encoder = LabelEncoder()
    df['mainroad'] = encoder.fit_transform(df['mainroad'])
    df['guestroom'] = encoder.fit_transform(df['guestroom'])
    df['basement'] = encoder.fit_transform(df['basement'])
    df['hotwaterheating'] = encoder.fit_transform(df['hotwaterheating'])
    df['airconditioning'] = encoder.fit_transform(df['airconditioning'])
    df['prefarea'] = encoder.fit_transform(df['prefarea'])

    df['furnishingstatus_furnished'] = (df['furnishingstatus'] ==
                                        'furnished')
    df['furnishingstatus_semi-furnished'] = (df['furnishingstatus'] ==
                                             'semi-furnished')
    df['furnishingstatus_unfurnished'] = (df['furnishingstatus'] ==
                                          'unfurnished')

    df = df.drop(columns=['furnishingstatus'])

    prediction = model.predict(df)

    assert prediction[0] == 4200000


test_model()
