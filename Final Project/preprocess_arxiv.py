import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to clean text
def clean_text(text):
    """
    Cleans the input text by:
    - Lowercasing the text
    - Removing special characters, numbers, and extra spaces
    Args:
        text (str): Input text (e.g., abstract)
    Returns:
        str: Cleaned text
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = " ".join(text.split())  # Remove extra spaces
    return text

def preprocess_data(input_csv_path, output_processed_path):
    """
    Preprocess the CSV dataset by cleaning text and encoding categories.
    Splits the data into training and test sets.
    
    Args:
        input_csv_path (str): Path to the input CSV file.
        output_processed_path (str): Path to save the processed data.
    """
    # Load dataset
    print(f"Loading data from {input_csv_path}...")
    data = pd.read_csv(input_csv_path)

    # Clean abstract text
    print("Cleaning text...")
    data['abstract'] = data['abstract'].apply(clean_text)

    # Encode categories into numerical labels
    print("Encoding categories...")
    label_encoder = LabelEncoder()
    data['category_encoded'] = label_encoder.fit_transform(data['categories'])

    # Save label mappings for future use
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Category mappings:", label_mapping)

    # Split into training and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data['abstract'], 
        data['category_encoded'], 
        test_size=0.2, 
        random_state=42
    )

    # Save the splits
    print(f"Saving processed data to {output_processed_path}...")
    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_mapping": label_mapping
    }
    pd.to_pickle(processed_data, output_processed_path)

    print("Data preprocessing complete!")

# Main entry point
if __name__ == "__main__":
    # Input CSV file path (from the extraction step)
    input_csv_path = r"C:\Users\jason\OneDrive - Rice University\Rice_U\ELEC 576 Intro to ML\Final project\arxiv_subset.csv"

    # Output file path for processed data
    output_processed_path = r"C:\Users\jason\OneDrive - Rice University\Rice_U\ELEC 576 Intro to ML\Final project\processed_data.pkl"

    # Run the preprocessing
    preprocess_data(input_csv_path, output_processed_path)
