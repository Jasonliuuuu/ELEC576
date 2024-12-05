import pandas as pd

def verify_processed_data(processed_file_path):
    """
    Verifies the processed data saved in a pickle file.
    Args:
        processed_file_path (str): Path to the pickle file containing processed data.
    """
    try:
        # Load the processed data
        print(f"Loading processed data from {processed_file_path}...")
        processed_data = pd.read_pickle(processed_file_path)

        # Display the keys in the processed data
        print("Keys in the processed data:", processed_data.keys())

        # Display a sample of each component
        print("\nSample X_train (abstracts):")
        print(processed_data['X_train'].head())

        print("\nSample y_train (encoded categories):")
        print(processed_data['y_train'].head())

        print("\nCategory Label Mapping:")
        print(processed_data['label_mapping'])

    except Exception as e:
        print(f"An error occurred while verifying processed data: {e}")

# Main entry point
if __name__ == "__main__":
    # Path to the processed data file
    processed_file_path = r"C:\Users\jason\OneDrive - Rice University\Rice_U\ELEC 576 Intro to ML\Final project\processed_data.pkl"
    
    # Run the verification
    verify_processed_data(processed_file_path)
