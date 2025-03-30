import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import urllib.request
import zipfile

# Constants
IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load pretrained model (automatically downloads if missing)
def load_pretrained_model():
    model_path = "emotion_model.h5"
    
    if not os.path.exists(model_path):
        # Download model if not found
        try:
            messagebox.showinfo("Info", "Downloading pretrained model...")
            
            # Download zip
            urllib.request.urlretrieve(MODEL_URL, "temp_model.zip")
            
            # Extract
            with zipfile.ZipFile("temp_model.zip", 'r') as zip_ref:
                zip_ref.extractall()
            
            os.remove("temp_model.zip")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download model: {str(e)}")
            return None
    
    try:
        model = load_model(model_path)
        print("Pretrained model loaded successfully")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Model loading failed: {str(e)}")
        return None

# Initialize GUI
root = tk.Tk()
root.geometry('800x600')
root.title('Emotion Detector')
root.configure(background='#CDCDCD')

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_pretrained_model()  # Load pretrained model

# GUI Components
result_label = tk.Label(root, background='#CDCDCD', font=('arial', 15, 'bold'))
image_label = tk.Label(root)

def detect_emotion(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            result_label.config(text="No face detected", foreground='red')
            return

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=(0, -1))
            
            predictions = emotion_model.predict(face)
            emotion_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            
            result_label.config(
                text=f"Emotion: {EMOTIONS[emotion_idx]}\nConfidence: {confidence:.2%}",
                foreground='green'
            )
            
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", foreground='red')

def upload_action():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not filepath:
        return
    
    try:
        img = Image.open(filepath)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text="")
        
        detect_emotion(filepath)  # Auto-detect on upload
        
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", foreground='red')

# GUI Layout
title = tk.Label(
    root, 
    text="Emotion Detector (Pretrained Model)", 
    pady=20, 
    font=('arial', 20, 'bold'),
    bg='#CDCDCD', 
    fg='#364156'
)
title.pack()

upload_btn = tk.Button(
    root, 
    text="Upload & Detect", 
    command=upload_action,
    padx=10, 
    pady=5,
    bg='#364156', 
    fg='white',
    font=('arial', 12, 'bold')
)
upload_btn.pack(pady=20)

image_label.pack(pady=20)
result_label.pack(pady=10)

root.mainloop()

# Note: The model URL and the model file name should be updated according to your actual model location and name.