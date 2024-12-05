import numpy as np

def inspect_saved_data(file_path):
    """
    Inspect the saved data from the prepared_data.npz file.
    
    Args:
        file_path (str): Path to the .npz file containing prepared data.
    """
    print(f"Loading data from {file_path}...")
    data = np.load(file_path)

    # Display shapes of the saved arrays
    print("Data contents:")
    print(f"X_train_padded shape: {data['X_train_padded'].shape}")
    print(f"X_test_padded shape: {data['X_test_padded'].shape}")
    print(f"Embedding matrix shape: {data['embedding_matrix'].shape}")

    # Optionally, display a small sample
    print("\nSample X_train_padded (first sequence):")
    print(data['X_train_padded'][0])

    print("\nSample from embedding matrix (first word vector):")
    print(data['embedding_matrix'][0])

if __name__ == "__main__":
    # Path to the prepared data file
    file_path = "prepared_data.npz"
    
    # Inspect the data
    inspect_saved_data(file_path)


#the goal of this cript is 
#1. Verigy the processed output
#2. Debugging and Validation
#3. Understand the Data