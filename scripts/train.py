import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

# --- Configuration ---
# File paths for the data and the final model
PROCESSED_DATA_PATH = 'data/'
CHAR_MAP_PATH = 'data/char_to_idx.json'
SAVED_MODEL_PATH = 'app/gender_predictor_model.keras' # Save directly to the app folder

# --- Model Hyperparameters ---
# These values can be tuned to improve model performance
EMBEDDING_DIM = 64      # Size of the vector for each character
LSTM_UNITS = 128        # Number of memory units in the LSTM layer
DROPOUT_RATE = 0.5      # Helps prevent overfitting by randomly dropping connections
EPOCHS = 10             # Number of times the model sees the entire training dataset
BATCH_SIZE = 256        # Number of samples to process in each batch

# Define the model architecture using Keras
Sequential = keras.models.Sequential
Embedding = keras.layers.Embedding
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout

def train_model():
    """
    Loads preprocessed data, builds, trains, and saves the neural network model.
    """
    print("Starting model training process...")

    # 1. Load the processed data
    print("Loading preprocessed data...")
    X_train = np.load(PROCESSED_DATA_PATH + 'X_train.npy')
    y_train = np.load(PROCESSED_DATA_PATH + 'y_train.npy')
    X_test = np.load(PROCESSED_DATA_PATH + 'X_test.npy')
    y_test = np.load(PROCESSED_DATA_PATH + 'y_test.npy')

    # Load the character-to-index mapping to get the vocabulary size
    with open(CHAR_MAP_PATH, 'r') as f:
        char_to_idx = json.load(f)
    
    vocab_size = len(char_to_idx) + 1 # +1 for the padding token (index 0)
    max_name_length = X_train.shape[1] # Get max length from the data shape

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Max Name Length: {max_name_length}")

    # 2. Build the model architecture
    print("Building the model...")
    model = Sequential([
        # The Embedding layer takes integer-encoded vocabulary and looks up the embedding vector for each.
        # It turns our sequence of numbers into a sequence of dense vectors.
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_name_length),

        # The LSTM layer processes the sequence of vectors, learning temporal patterns.
        LSTM(units=LSTM_UNITS),

        # Dropout is a regularization technique to prevent overfitting.
        # It randomly sets a fraction of input units to 0 at each update during training.
        Dropout(DROPOUT_RATE),

        # The final Dense layer produces the output prediction.
        # We use 1 unit for binary classification and a 'sigmoid' activation function
        # to output a probability between 0 and 1.
        Dense(1, activation='sigmoid')
    ])

    # 3. Compile the model
    # This configures the model for training.
    model.compile(
        optimizer='adam',                   # Adam is an efficient and popular optimization algorithm.
        loss='binary_crossentropy',        # For binary (0/1) classification problems.
        metrics=['accuracy']               # We want to monitor the accuracy during training.
    )

    # Print a summary of the model's architecture
    model.summary()

    # 4. Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test), # We use the test set as a validation set
        verbose=1                          # Show progress bar
    )

    # 5. Evaluate the model on the test set
    print("\nEvaluating the model on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # 6. Save the trained model
    print(f"Saving the trained model to {SAVED_MODEL_PATH}...")
    model.save(SAVED_MODEL_PATH)
    print("Model training complete and model saved.")

if __name__ == '__main__':
    train_model()
