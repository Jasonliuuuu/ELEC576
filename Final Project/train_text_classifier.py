import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split

# Load the prepared data
print("Loading data from prepared_data.npz...")
data = np.load("prepared_data.npz")
X_train_padded = data["X_train_padded"]
X_test_padded = data["X_test_padded"]
y_train = data["y_train"]
y_test = data["y_test"]
embedding_matrix = data["embedding_matrix"]
num_classes = 76121  # Define the number of output classes

# Validate and correct labels
print("Validating and correcting labels...")
print(f"Original labels: Max={np.max(y_train)}, Min={np.min(y_train)}")
y_train = np.clip(y_train, 0, num_classes - 1)
y_test = np.clip(y_test, 0, num_classes - 1)
print(f"Corrected labels: Max={np.max(y_train)}, Min={np.min(y_train)}")

# Verify label ranges
if np.max(y_train) >= num_classes or np.min(y_train) < 0:
    raise ValueError("Labels are out of range after correction. Check the dataset.")

# Split training data into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_padded, y_train, test_size=0.1, random_state=42)

# Build the model
print("Building the model...")
model = Sequential([
    Embedding(input_dim=embedding_matrix.shape[0],
              output_dim=embedding_matrix.shape[1],
              weights=[embedding_matrix],
              input_length=X_train_padded.shape[1],
              trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model summary
model.summary()

# Train the model
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=128
)

# Evaluate the model on the test set
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, batch_size=128)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the model
model.save("text_classifier_model.h5")
print("Model saved to text_classifier_model.h5.")
