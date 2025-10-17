# Model Directory

## ⚠️ Important

Place your trained TensorFlow model file here with the exact name: `model.h5`

## Model Requirements

- **Format**: Keras/TensorFlow H5 format (`.h5`)
- **File name**: Must be named exactly `model.h5`
- **Input**: The model should accept a 1D array of numerical values
- **Output**: Binary classification (0 or 1), where:
  - Output > 0.5 → "Closed" (drowsy/eyes closed)
  - Output ≤ 0.5 → "Open" (alert/eyes open)

## Training Your Model

If you haven't trained your model yet, you'll need to:

1. Prepare your drowsiness detection dataset
2. Train your model using TensorFlow/Keras
3. Save the model using: `model.save('model.h5')`
4. Place the saved `model.h5` file in this directory

## Example Model Structure

```python
import tensorflow as tf
from tensorflow import keras

# Example model architecture
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(num_features,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# After training
model.save('model.h5')
```

## Note

The backend will start without the model file, but the `/predict` endpoint will return an error until `model.h5` is present in this directory.

