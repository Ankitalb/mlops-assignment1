from flask import Flask, jsonify, request

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return "Welcome to the Flask app!"

# Define a route to return some data (JSON format)
@app.route('/data', methods=['GET'])
def get_data():
    data = {"message": "Hello from Flask!"}
    return jsonify(data)

# Define a route that accepts POST requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the JSON data from the request
    prediction = {"prediction": "Your model prediction here"}  # Just an example
    return jsonify(prediction)

if __name__ == '__main__':
    # Run the app on localhost at port 5000
    app.run(debug=True)
