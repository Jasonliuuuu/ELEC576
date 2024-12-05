import numpy as np

def verify_prepared_data(file_path):
    """
    Verify the contents of the prepared data file.
    
    Args:
        file_path (str): Path to the .npz file.
    """
    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    
    # List all keys in the .npz file
    print("Keys in prepared data:", list(data.keys()))
    
    # Optionally, check the shapes of the arrays
    for key in data.keys():
        print(f"{key} shape: {data[key].shape}")

if __name__ == "__main__":
    # Path to the prepared data file
    file_path = "prepared_data.npz"
    
    # Verify the contents
    verify_prepared_data(file_path)
