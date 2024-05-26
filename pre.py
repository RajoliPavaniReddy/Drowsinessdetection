import numpy as np
import os
from PIL import Image
from keras.preprocessing import image

data_dir = r"C:\Users\pavan\OneDrive\Desktop\bavaksh\train"
categories = ['yawn', 'no_yawn', 'Open', 'Closed']



# Function to preprocess an image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((100, 100))  # Adjust size if needed
    img_array = np.array(img)
    return img_array / 255.0  # Normalize pixel values



for category in categories:
    sequences = []
    labels = []
    processed_images_count = 0
    
    folder_path = os.path.join(data_dir, category)
    class_label = categories.index(category)
    
    # Preprocess images and append to sequences and labels lists
    for file_name in os.listdir(folder_path):
        if file_name.startswith('.'):  # Skip hidden files
            continue
        if os.path.isdir(os.path.join(folder_path, file_name)):  # Skip subdirectories
            continue
        try:
            img_array = preprocess_image(os.path.join(folder_path, file_name))
            sequences.append(img_array)
            labels.append(class_label)
            processed_images_count += 1
        except Exception as e:
            print(f"Error processing {os.path.join(folder_path, file_name)}: {e}")

    # Convert sequences and labels to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Save the preprocessed data for each category
    np.save(f'preprocessed_sequences_{category}.npy', sequences)
    np.save(f'preprocessed_labels_{category}.npy', labels)

