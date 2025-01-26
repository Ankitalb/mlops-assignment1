import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_URL = 'C:/Users/ankita/Documents/mlops/Assignement1/data/dataset.csv'
data = pd.read_csv(DATA_URL)
df = pd.DataFrame(data)

encoder = LabelEncoder()
df['mainroad'] = encoder.fit_transform(df['mainroad'])
df['guestroom'] = encoder.fit_transform(df['guestroom'])
df['basement'] = encoder.fit_transform(df['basement'])
df['hotwaterheating'] = encoder.fit_transform(df['hotwaterheating'])
df['airconditioning'] = encoder.fit_transform(df['airconditioning'])
df['prefarea'] = encoder.fit_transform(df['prefarea'])
df = pd.get_dummies(df, columns=['furnishingstatus'])

X = df.drop("price", axis=1)
y = df['price']
print(X)
print(y)
print(y.shape)
# If y is a 2D array (e.g., shape (n_samples, 1)), reshape it to 1D
y = y.to_numpy()
print(y.shape)  # This should now be (n_samples,)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)


def grid_search_with_mlflow(X_train, y_train, X_test, y_test):
    # Hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                               n_jobs=-1, verbose=2)

    # Start the MLflow experiment run
    with mlflow.start_run():
        # Perform grid search
        grid_search.fit(X_train, y_train)

        # Log the best hyperparameters found during the grid search
        mlflow.log_params(grid_search.best_params_)
        # Log the best score
        mlflow.log_metric("best_accuracy", grid_search.best_score_)

        # Log the best model
        mlflow.sklearn.log_model(grid_search.best_estimator_,
                                 "random_forest_model2")

        # Print the best parameters and the best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-validation Accuracy:", grid_search.best_score_)

        # Evaluate the best model on the test set
        y_pred = grid_search.best_estimator_.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        # Log the test accuracy
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(mlflow.get_tracking_uri())

        print(f"Test Accuracy: {test_accuracy}")


# Perform GridSearchCV with MLflow logging
grid_search_with_mlflow(X_train, y_train, X_test, y_test)
