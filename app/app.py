import flask
from flask import request, jsonify
import numpy as np
import tensorflow as tf
import json

# --- Configuration & Globals ---
# Create the Flask application object
app = flask.Flask(__name__)

# Define paths for the model and the character map
MODEL_PATH = 'gender_predictor_model.keras'
CHAR_MAP_PATH = 'char_to_idx.json'
MAX_NAME_LENGTH = 15 # This must match the value from the training script

# --- Model & Data Loading ---
# Load the trained Keras model
# We do this once when the app starts to avoid reloading on every request
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("* Model loaded successfully")
except Exception as e:
    print(f"* Error loading model: {e}")
    model = None

# Load the character-to-index mapping
try:
    with open(CHAR_MAP_PATH, 'r') as f:
        char_to_idx = json.load(f)
    print("* Character map loaded successfully")
except Exception as e:
    print(f"* Error loading character map: {e}")
    char_to_idx = None

# --- Helper Function ---
def preprocess_name(name):
    """
    Converts a single name string into a padded sequence of integers
    that can be fed into the model.
    """
    # Convert name to lowercase and create a sequence of character indices
    seq = [char_to_idx.get(char, 0) for char in name.lower()] # Use 0 for unknown chars
    # Pad the sequence to the fixed length
    padded_seq = seq[:MAX_NAME_LENGTH] + [0] * (MAX_NAME_LENGTH - len(seq))
    
    # Reshape for the model (needs a batch dimension)
    return np.array([padded_seq])

# --- API Endpoints ---
@app.route("/")
def index():
    """A simple endpoint to test if the server is running."""
    return "<h1>Gender Predictor API</h1><p>Send a POST request to /predict</p>"

@app.route("/predict", methods=['POST'])
def predict():
    """
    The main prediction endpoint.
    Expects a JSON payload with a "name" key.
    e.g., curl -X POST -H "Content-Type: application/json" -d '{"name": "alex"}' http://127.0.0.1:5000/predict
    """
    # Ensure the model and char_map are loaded
    if model is None or char_to_idx is None:
        return jsonify({"error": "Model or character map not loaded. Check server logs."}), 500

    # Get the JSON data from the request
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({"error": "Invalid input. Please provide a 'name' in the JSON payload."}), 400

    name = data['name']

    # Preprocess the input name
    processed_name = preprocess_name(name)

    # Make a prediction using the loaded model
    prediction_prob = model.predict(processed_name)[0][0]

    # Interpret the prediction
    # Our model outputs a probability. We'll say > 0.5 is Female.
    gender = "Female" if prediction_prob > 0.5 else "Male"

    # Create the response JSON
    response = {
        'name': name,
        'gender': gender,
        'probability': float(prediction_prob)
    }

    return jsonify(response)

# --- Main ---
if __name__ == '__main__':
    # Start the Flask development server
    # Use 0.0.0.0 to make it accessible from outside the container later
    app.run(host='0.0.0.0', port=5000, debug=True)
