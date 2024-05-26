import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load preprocessed data
categories = ['yawn', 'no_yawn', 'Open', 'Closed']
preprocessed_sequences = []
preprocessed_labels = []

for category in categories:
    sequences = np.load(f'preprocessed_sequences_{category}.npy')
    labels = np.load(f'preprocessed_labels_{category}.npy')
    preprocessed_sequences.append(sequences)
    preprocessed_labels.append(labels)

# Concatenate sequences and labels
sequences = np.concatenate(preprocessed_sequences, axis=0)
labels = np.concatenate(preprocessed_labels, axis=0)

# Shuffle data
indices = np.arange(len(sequences))
np.random.shuffle(indices)
sequences = sequences[indices]
labels = labels[indices]

# Split data into train and test sets
split_ratio = 0.8
split_index = int(len(sequences) * split_ratio)
train_sequences, test_sequences = sequences[:split_index], sequences[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(categories), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_sequences, train_labels, epochs=10, validation_split=0.2)

# Evaluate model
test_loss, test_acc = model.evaluate(test_sequences, test_labels)
print('Test accuracy:', test_acc)
# Save the model
model.save('cnn_model.h5')
