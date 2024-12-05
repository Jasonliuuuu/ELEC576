import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_glove_embeddings(file_path, embedding_dim=100):
    """
    Load GloVe embeddings from a .txt file and create a word-to-vector dictionary.
    
    Args:
        file_path (str): Path to the GloVe .txt file.
        embedding_dim (int): Dimensionality of the word vectors.
    
    Returns:
        dict: A dictionary mapping words to their GloVe vector representations.
    """
    print(f"Loading GloVe embeddings from {file_path}...")
    embeddings_index = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index


def prepare_embedding_matrix(tokenizer, embeddings_index, embedding_dim=100):
    """
    Prepare the embedding matrix for the Keras Embedding layer.
    
    Args:
        tokenizer (Tokenizer): Tokenizer object fitted on the training data.
        embeddings_index (dict): GloVe embeddings as a word-to-vector dictionary.
        embedding_dim (int): Dimensionality of the word vectors.
    
    Returns:
        np.ndarray: Embedding matrix with shape (vocab_size, embedding_dim).
    """
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1  # Including padding token
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words found in GloVe
            embedding_matrix[i] = embedding_vector
        else:
            # Words not found in GloVe (use random initialization)
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))
    
    print(f"Embedding matrix created with shape: {embedding_matrix.shape}")
    return embedding_matrix


def process_and_tokenize_texts(X_train, X_test, max_sequence_length=100):
    """
    Tokenize and pad text data for input to the model.
    
    Args:
        X_train (list): List of training text data (abstracts).
        X_test (list): List of test text data (abstracts).
        max_sequence_length (int): Maximum sequence length for padding.
    
    Returns:
        tuple: (X_train_padded, X_test_padded, tokenizer)
    """
    # Tokenize the texts
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    
    # Convert texts to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to uniform length
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding="post")
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding="post")
    
    print(f"Text data tokenized and padded. Training data shape: {X_train_padded.shape}")
    return X_train_padded, X_test_padded, tokenizer


if __name__ == "__main__":
    # File paths
    glove_path = r"C:\Users\jason\OneDrive - Rice University\Rice_U\ELEC 576 Intro to ML\Final project\glove.6B.100d.txt"
    processed_data_path = r"C:\Users\jason\OneDrive - Rice University\Rice_U\ELEC 576 Intro to ML\Final project\processed_data.pkl"
    
    # Parameters
    embedding_dim = 100
    max_sequence_length = 100  # Adjust based on dataset and model
    
    # Load processed data
    print(f"Loading processed data from {processed_data_path}...")
    processed_data = pd.read_pickle(processed_data_path)
    X_train = processed_data["X_train"]
    X_test = processed_data["X_test"]
    y_train = processed_data["y_train"]
    y_test = processed_data["y_test"]
    
    # Load GloVe embeddings
    embeddings_index = load_glove_embeddings(glove_path, embedding_dim)
    
    # Tokenize and pad texts
    X_train_padded, X_test_padded, tokenizer = process_and_tokenize_texts(X_train, X_test, max_sequence_length)
    
    # Prepare embedding matrix
    embedding_matrix = prepare_embedding_matrix(tokenizer, embeddings_index, embedding_dim)
    
    # Save prepared data including y_train and y_test
    np.savez_compressed("prepared_data.npz", 
                        X_train_padded=X_train_padded, 
                        X_test_padded=X_test_padded, 
                        y_train=y_train, 
                        y_test=y_test, 
                        embedding_matrix=embedding_matrix)
    print("Prepared data saved to 'prepared_data.npz'.")
