import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

# Constants
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 15
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create and train model
def create_and_train_model(data_path):
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Model architecture
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(64, (5,5), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1
    )
    
    return model

# Initialize Tkinter
root = tk.Tk()
root.geometry('800x600')
root.title('Emotion Detector')
root.configure(background='#CDCDCD')

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Global variables
emotion_model = None
result_label = tk.Label(root, background='#CDCDCD', font=('arial', 15, 'bold'))
image_label = tk.Label(root)

def detect_emotion(image_path):
    global emotion_model
    try:
        if emotion_model is None:
            result_label.config(text="Please train or load a model first", foreground='red')
            return
            
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
        
        detect_btn = tk.Button(
            root, 
            text="Detect Emotion", 
            command=lambda: detect_emotion(filepath),
            bg='#364156', 
            fg='white',
            font=('arial', 10, 'bold')
        )
        detect_btn.place(relx=0.5, rely=0.9, anchor=tk.CENTER)
        
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", foreground='red')

def train_model():
    global emotion_model
    data_path = filedialog.askdirectory(title="Select Training Data Directory")
    if not data_path:
        return
    
    try:
        result_label.config(text="Training model...", foreground='blue')
        root.update()
        
        emotion_model = create_and_train_model(data_path)
        emotion_model.save("emotion_model.h5")
        messagebox.showinfo("Success", "Model trained and saved successfully!")
        
    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {str(e)}")
    finally:
        result_label.config(text="")

# GUI Layout
title = tk.Label(
    root, 
    text="Emotion Detector", 
    pady=20, 
    font=('arial', 20, 'bold'),
    bg='#CDCDCD', 
    fg='#364156'
)
title.pack()

train_btn = tk.Button(
    root, 
    text="Train New Model", 
    command=train_model,
    padx=10, 
    pady=5,
    bg='#364156', 
    fg='white',
    font=('arial', 10, 'bold')
)
train_btn.pack(pady=10)

upload_btn = tk.Button(
    root, 
    text="Upload Image", 
    command=upload_action,
    padx=10, 
    pady=5,
    bg='#364156', 
    fg='white',
    font=('arial', 10, 'bold')
)
upload_btn.pack(pady=10)

image_label.pack(pady=20)
result_label.pack(pady=10)

root.mainloop() 

# Note: use train dataset for upload model as the weights and model were not compatible with the GUI.