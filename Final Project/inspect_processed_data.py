import pandas as pd

def inspect_processed_data(processed_file_path):
    """
    Inspects the processed data saved in a pickle file.
    Displays sample data and verifies the training and test splits.
    
    Args:
        processed_file_path (str): Path to the pickle file containing processed data.
    """
    try:
        # Load the processed data
        print(f"Loading processed data from {processed_file_path}...")
        processed_data = pd.read_pickle(processed_file_path)

        # Display the keys in the processed data
        print("\nKeys in the processed data:", processed_data.keys())

        # Display sample training data
        print("\nSample Training Data (X_train - Abstracts):")
        print(processed_data['X_train'].head())

        print("\nSample Training Labels (y_train - Encoded Categories):")
        print(processed_data['y_train'].head())

        # Display the category label mapping
        print("\nCategory Label Mapping:")
        for category, encoded_label in processed_data['label_mapping'].items():
            print(f"{encoded_label}: {category}")

        # Check dataset sizes
        print(f"\nTraining Set Size: {len(processed_data['X_train'])}")
        print(f"Test Set Size: {len(processed_data['X_test'])}")

        # Verify sample test data
        print("\nSample Test Data (X_test - Abstracts):")
        print(processed_data['X_test'].head())

        print("\nSample Test Labels (y_test - Encoded Categories):")
        print(processed_data['y_test'].head())

    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_file_path}")
    except Exception as e:
        print(f"An error occurred during inspection: {e}")

# Main entry point
if __name__ == "__main__":
    # Path to the processed data file
    processed_file_path = r"C:\Users\jason\OneDrive - Rice University\Rice_U\ELEC 576 Intro to ML\Final project\processed_data.pkl"

    # Run the inspection
    inspect_processed_data(processed_file_path)
    
    #Now that the data is clean, split, and ready, the next step is to vectorize the text data and train a machine learning or deep learning model.
