import tkinter as tk
from tkinter import filedialog, Label, Button
import os
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.cnn_model = load_model('cnn_model.h5')
        self.cnn_lstm_model = load_model('drowsiness_detection_cnn_lstm_model.h5')
        self.categories = ['yawn', 'no_yawn', 'Open', 'Closed']
        self.image_label = Label(self.root, text="Selected Image Will Appear Here")
        self.image_label.pack()
        self.predictions_label = Label(self.root, text="")
        self.predictions_label.pack()
        self.select_button = Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack()

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')  # Convert image to grayscale
        img = img.resize((100, 100))  # Adjust size if needed
        img_array = np.array(img)
        return img_array.reshape(-1, 100, 100, 1) / 255.0  # Normalize pixel values and reshape

    def predict_cnn(self, image_array):
        prediction = self.cnn_model.predict(image_array)
        return np.argmax(prediction)

    def predict_cnn_lstm(self, image_array):
        prediction = self.cnn_lstm_model.predict(image_array)
        return np.argmax(prediction)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", ".jpg;.png"), ("All files", ".")]
        )
        if file_path:
            img = Image.open(file_path)
            img = img.resize((200, 200))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            img_array = self.preprocess_image(file_path)
            cnn_prediction = self.predict_cnn(img_array)
            cnn_lstm_prediction = self.predict_cnn_lstm(img_array)
            cnn_prediction_label = self.categories[cnn_prediction]
            cnn_lstm_prediction_label = self.categories[cnn_lstm_prediction]
            self.predictions_label.config(text=f"CNN Prediction: {cnn_prediction_label}\nCNN-LSTM Prediction: {cnn_lstm_prediction_label}")


root = tk.Tk()
app = ImageClassifierApp(root)
root.mainloop()