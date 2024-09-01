import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError

# Load the model with the mae custom object
loaded_model = load_model("final_model.h5", custom_objects={'mae': MeanAbsoluteError()})

def custom_gender_prediction(age, gender_pred):
    if 20 <= age <= 30:
        return 0 if gender_pred < 0.5 else 1  # Female if gender_pred < 0.5, else Male
    else:
        return int(np.round(gender_pred))

def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (48, 48))
        image_normalized = image_resized / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)
        
        gender_pred, age_pred = loaded_model.predict(image_input)
        
        final_gender = custom_gender_prediction(age_pred[0][0], gender_pred[0][0])
        gender_text = 'Female' if final_gender == 0 else 'Male'
        
        result_text.set(f"Predicted Gender: {gender_text}\nPredicted Age: {int(age_pred[0][0])}")

        img = Image.open(file_path)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

# GUI setup
root = tk.Tk()
root.title("Gender and Age Detection")
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text)
result_label.pack()
panel = tk.Label(root)
panel.pack()
btn = tk.Button(root, text="Load Image and Predict", command=load_and_predict_image)
btn.pack()
root.mainloop()
