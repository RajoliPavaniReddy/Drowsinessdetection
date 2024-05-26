import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape
from sklearn.model_selection import train_test_split

# Load preprocessed data for each category
categories = ['yawn', 'no_yawn', 'Open', 'Closed']

# Define a function to load preprocessed data
def load_data(category):
    sequences = np.load(f'preprocessed_sequences_{category}.npy')
    labels = np.load(f'preprocessed_labels_{category}.npy')
    return sequences, labels

# Concatenate sequences and labels for all categories
all_sequences = []
all_labels = []
for category in categories:
    sequences, labels = load_data(category)
    all_sequences.extend(sequences)
    all_labels.extend(labels)

# Convert lists to numpy arrays
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_sequences, all_labels, test_size=0.2, random_state=42)

# Reshape the data for convolutional layers
X_train = X_train.reshape(-1, 100, 100, 1)  # Assuming input shape is (100, 100)
X_test = X_test.reshape(-1, 100, 100, 1)

# Define the CNN-LSTM model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))  # Added dense layer after flatten
model.add(Dropout(0.5))  # Add dropout
model.add(Reshape((-1, 128)))  # Reshape for LSTM input
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(64))
model.add(Dropout(0.1))
model.add(Dense(len(categories), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Print accuracy
_, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save('drowsiness_detection_cnn_lstm_model.h5')
