# 🧠 Drowsiness Detection Model Training

This repository contains the code to train a CNN (Convolutional Neural Network) model for drowsiness detection based on eye state classification.

## 📋 Overview

The model classifies eye states into two categories:
- **Open** (Person is alert)
- **Closed** (Person is drowsy)

## 🏗️ Model Architecture

```
Input: 24x24 Grayscale Image
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (1x1)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (1x1)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (1x1)
    ↓
Dropout (25%)
    ↓
Flatten
    ↓
Dense (128 units) + ReLU
    ↓
Dropout (50%)
    ↓
Dense (2 units) + Softmax
    ↓
Output: [Closed Probability, Open Probability]
```

## 📂 Dataset Structure

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── Open/           # Training images of open eyes
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── Closed/         # Training images of closed eyes
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── valid/
    ├── Open/           # Validation images of open eyes
    │   ├── img1.jpg
    │   └── ...
    └── Closed/         # Validation images of closed eyes
        ├── img1.jpg
        └── ...
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow keras numpy matplotlib
```

### 2. Prepare Your Dataset

- Collect images of open and closed eyes
- Split into training and validation sets (e.g., 80/20 split)
- Organize in the structure shown above

### 3. Train the Model

```bash
python train_model.py
```

## ⚙️ Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | 32 | Number of images per training batch |
| **Image Size** | 24x24 | Input dimensions (grayscale) |
| **Epochs** | 15 | Number of training iterations |
| **Optimizer** | Adam | Adaptive learning rate optimizer |
| **Loss Function** | Categorical Crossentropy | For multi-class classification |

## 📊 Training Output

After training, you'll get:

1. **model/eye_state_model.h5** - Trained model file
2. **model/training_history.png** - Accuracy and loss plots
3. **Console output** - Training progress and final metrics

Example output:
```
Steps per epoch: 250, Validation steps: 50

Model Architecture:
============================================================
...
============================================================

Starting training...
Epoch 1/15
250/250 [==============================] - 45s 180ms/step
...
Epoch 15/15
250/250 [==============================] - 42s 168ms/step

✅ Model saved to: model/eye_state_model.h5
✅ Training history plot saved to: model/training_history.png

Final Performance:
============================================================
Training Accuracy: 0.9678
Validation Accuracy: 0.9423
============================================================
```

## 🎯 Model Performance Tips

To improve model accuracy:

1. **More Data**: Collect more diverse images
2. **Data Augmentation**: Add rotation, flipping, brightness changes
3. **Hyperparameter Tuning**: Adjust learning rate, batch size, epochs
4. **Regularization**: Modify dropout rates
5. **Architecture**: Try different layer configurations

## 📦 Saved Model

The trained model is saved as **`model/eye_state_model.h5`** and can be loaded for inference:

```python
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model('model/eye_state_model.h5')

# Prepare an image
img = Image.open('test_eye.jpg').convert('L')  # Grayscale
img = img.resize((24, 24))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=(0, -1))

# Predict
prediction = model.predict(img_array)
print(f"Closed: {prediction[0][0]:.4f}, Open: {prediction[0][1]:.4f}")

if prediction[0][1] > prediction[0][0]:
    print("Eyes are OPEN (Alert)")
else:
    print("Eyes are CLOSED (Drowsy)")
```

## 🔬 Model Details

- **Input Shape**: (24, 24, 1) - 24x24 grayscale image
- **Output Shape**: (2,) - Two probabilities [Closed, Open]
- **Total Parameters**: ~200K trainable parameters
- **File Size**: ~1.8 MB

## 📚 Dataset Recommendations

### Public Datasets:
- **MRL Eye Dataset**: http://mrl.cs.vsb.cz/eyedataset
- **CEW (Closed Eyes in the Wild)**: http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html
- **YawDD (Yawn Detection Dataset)**: Contains eye state labels

### Custom Dataset Tips:
- Minimum 1000 images per class (Open/Closed)
- Include various lighting conditions
- Multiple people of different ages/ethnicities
- Different eye shapes and sizes
- With/without glasses

## 🛠️ Troubleshooting

### Low Accuracy
- Increase dataset size
- Add data augmentation
- Train for more epochs
- Reduce dropout if overfitting isn't an issue

### Out of Memory
- Reduce batch size
- Use smaller image size
- Close other applications

### Slow Training
- Use GPU if available
- Reduce image size
- Decrease batch size

## 📄 License

MIT License - Feel free to use for your projects!

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Share your trained models
- Submit pull requests

---

**Built with TensorFlow & Keras** 🧠
