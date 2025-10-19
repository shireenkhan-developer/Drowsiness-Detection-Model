"""
Drowsiness Detection Model Training Script
==========================================

This script trains a CNN model to detect eye states (Open/Closed) for drowsiness detection.

Dataset Structure Required:
---------------------------
data/
├── train/
│   ├── Open/      # Training images of open eyes
│   └── Closed/    # Training images of closed eyes
└── valid/
    ├── Open/      # Validation images of open eyes
    └── Closed/    # Validation images of closed eyes

Model Architecture:
------------------
- Conv2D layers for feature extraction
- MaxPooling for dimensionality reduction
- Dropout for regularization
- Dense layers for classification
- Output: Binary classification (Open/Closed)

Requirements:
------------
pip install tensorflow keras numpy matplotlib

Usage:
------
python train_model.py
"""

import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random, shutil
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    """
    Creates an image data generator for training/validation
    
    Args:
        dir: Directory containing class subdirectories
        gen: ImageDataGenerator instance
        shuffle: Whether to shuffle the data
        batch_size: Number of images per batch
        target_size: Target size for images (width, height)
        class_mode: Type of label arrays (categorical for 2+ classes)
    
    Returns:
        DirectoryIterator yielding batches of images and labels
    """
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale',
        class_mode=class_mode,
        target_size=target_size
    )


# Hyperparameters
BS = 32  # Batch size
TS = (24, 24)  # Target image size

# Create data generators
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)

# Calculate steps per epoch
SPE = len(train_batch.classes) // BS  # Steps per epoch
VS = len(valid_batch.classes) // BS   # Validation steps
print(f"Steps per epoch: {SPE}, Validation steps: {VS}")


# Build CNN Model
model = Sequential([
    # First convolutional block
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    
    # Second convolutional block
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    
    # Third convolutional block (64 filters)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    
    # Dropout for regularization
    Dropout(0.25),
    
    # Flatten to 1D
    Flatten(),
    
    # Fully connected layer
    Dense(128, activation='relu'),
    
    # Another dropout
    Dropout(0.5),
    
    # Output layer (2 classes: Open, Closed)
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print("\nModel Architecture:")
print("=" * 60)
model.summary()
print("=" * 60)

# Train the model
print("\nStarting training...")
history = model.fit_generator(
    train_batch,
    validation_data=valid_batch,
    epochs=15,
    steps_per_epoch=SPE,
    validation_steps=VS
)

# Save the trained model
os.makedirs('model', exist_ok=True)
model.save('model/eye_state_model.h5', overwrite=True)
print("\n✅ Model saved to: model/eye_state_model.h5")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model/training_history.png')
print("✅ Training history plot saved to: model/training_history.png")

# Evaluate final performance
print("\nFinal Performance:")
print("=" * 60)
train_loss, train_acc = model.evaluate_generator(train_batch, steps=SPE)
val_loss, val_acc = model.evaluate_generator(valid_batch, steps=VS)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print("=" * 60)

