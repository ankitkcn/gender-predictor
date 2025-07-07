import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Define file paths for input and output
RAW_DATA_PATH = 'data/NationalNames.csv'
PROCESSED_DATA_PATH = 'data/'
CHAR_MAP_PATH = 'data/char_to_idx.json'

# Define the maximum length for a name. Names longer than this will be truncated,
# and shorter ones will be padded.
MAX_NAME_LENGTH = 15

def preprocess_data():
    """
    Loads the raw data, processes it, and saves the results.
    Processing steps:
    1. Load the CSV file.
    2. Clean and aggregate data: get unique names and their most common gender.
    3. Create a character vocabulary and a char-to-index mapping.
    4. Convert names into sequences of integer indices.
    5. Pad sequences to a fixed length (MAX_NAME_LENGTH).
    6. Encode gender labels to 0 and 1.
    7. Split the data into training and testing sets.
    8. Save the processed data and the character map.
    """
    print("Starting data preprocessing...")

    # 1. Load the data
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows of data.")

    # 2. Clean and aggregate data
    # We only care about the name and the most likely gender.
    # We'll group by name and find the gender with the highest total count.
    name_gender = df.groupby('Name')['Gender'].agg(lambda x: x.value_counts().idxmax()).reset_index()
    print(f"Found {len(name_gender)} unique names.")

    # 3. Create character vocabulary
    # Get all unique characters from all names
    all_chars = set(''.join(name_gender['Name']))
    # Create a mapping from character to a unique integer index.
    # We reserve index 0 for padding.
    char_to_idx = {char: i + 1 for i, char in enumerate(sorted(all_chars))}
    vocab_size = len(char_to_idx) + 1  # +1 for the padding token
    print(f"Vocabulary size: {vocab_size} characters.")

    # 4. Convert names to sequences of integers
    name_sequences = []
    for name in name_gender['Name']:
        seq = [char_to_idx.get(char, 0) for char in name] # Use 0 for unknown chars, though we have all
        name_sequences.append(seq)

    # 5. Pad sequences
    # We use 'post' padding, meaning we add zeros at the end.
    padded_sequences = np.array([seq[:MAX_NAME_LENGTH] + [0] * (MAX_NAME_LENGTH - len(seq)) for seq in name_sequences])
    print(f"Padded sequences to max length of {MAX_NAME_LENGTH}.")

    # 6. Encode labels
    # Convert 'F' to 1 and 'M' to 0
    labels = np.array([1 if gender == 'F' else 0 for gender in name_gender['Gender']])

    # 7. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Split data into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # 8. Save processed data
    np.save(PROCESSED_DATA_PATH + 'X_train.npy', X_train)
    np.save(PROCESSED_DATA_PATH + 'y_train.npy', y_train)
    np.save(PROCESSED_DATA_PATH + 'X_test.npy', X_test)
    np.save(PROCESSED_DATA_PATH + 'y_test.npy', y_test)

    # Save the character-to-index mapping as a JSON file
    with open(CHAR_MAP_PATH, 'w') as f:
        json.dump(char_to_idx, f)

    print("Preprocessing complete. Processed files saved to 'data/' directory.")

if __name__ == '__main__':
    # This block runs when the script is executed directly
    print("entered preprocess.py")
    preprocess_data()
