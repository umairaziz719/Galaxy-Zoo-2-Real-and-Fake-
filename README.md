# Galaxy Zoo 2 - CNN-based Classification

Welcome to the **Galaxy Zoo 2** repository! This project leverages Convolutional Neural Networks (CNNs) to classify galaxies based on their morphological features. The dataset comes from the Galaxy Zoo 2 project, which involves citizen scientists classifying galaxies using images from the Sloan Digital Sky Survey (SDSS).

### Table of Contents

- [Introduction](#introduction)
- [Project Setup](#project-setup)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Making Predictions](#making-predictions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

**Galaxy Zoo 2** is a citizen science project that helps astronomers classify galaxies based on their shapes, sizes, and other visual properties. In this project, we use deep learning, specifically Convolutional Neural Networks (CNNs), to automate the classification process and predict galaxy morphologies.

We have trained a CNN model on the Galaxy Zoo 2 dataset, which includes labeled images of galaxies and their corresponding classification labels. This repository contains code for preprocessing the data, training a CNN model, evaluating its performance, and using it to make predictions on new data.

---

## Project Setup

### Requirements

Before you begin, ensure that you have the following dependencies installed:

- Python 3.x
- TensorFlow or PyTorch (depending on your preferred framework)
- `numpy` for numerical operations
- `matplotlib` for data visualization
- `pandas` for data manipulation
- `scikit-learn` for model evaluation
- `opencv-python` (for image processing)

You can install these dependencies using:

```bash
pip install -r requirements.txt
```

### Installing Dependencies

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/galaxy-zoo-2.git
cd galaxy-zoo-2
pip install -r requirements.txt
```

---

## Usage

### Preprocessing the Data

Before training the model, you need to load and preprocess the galaxy image data. Here’s how you can load and preprocess the images:

```python
import pandas as pd
import cv2
import numpy as np

# Load the Galaxy Zoo 2 dataset (CSV file with image paths and labels)
data = pd.read_csv('data/galaxy_zoo_2.csv')

# Example function to load and preprocess images
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)  # Read image from file
        img = cv2.resize(img, (128, 128))  # Resize to a fixed size
        img = img / 255.0  # Normalize pixel values
        images.append(img)
    return np.array(images)

# Preprocess the images
X = preprocess_images(data['image_path'])
y = data['label']  # Assuming labels are in the 'label' column
```

### Model Training

Now you can train the CNN model. Below is an example using TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (change if multiclass)
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
```

This model uses two convolutional layers followed by max-pooling, followed by a fully connected layer and an output layer for classification. If your problem is multi-class (e.g., predicting multiple galaxy types), you can adjust the output layer and loss function accordingly.

### Model Evaluation

After training, evaluate your model on a test set (or validation data) to see how well it performs:

```python
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

### Making Predictions

You can use the trained CNN model to classify new galaxy images:

```python
def predict_galaxy_class(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize the image
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = model.predict(img)
    return prediction

# Example usage
image_path = 'new_galaxy_image.jpg'
prediction = predict_galaxy_class(image_path)
print(f"Predicted class: {prediction}")
```

---

## Contributing

We welcome contributions from the community! If you want to contribute, follow these steps:

1. Fork the repository to your own GitHub account.
2. Create a new branch (`git checkout -b my-feature`).
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork (`git push origin my-feature`).
5. Open a Pull Request to merge your changes into the `main` branch.

Before contributing, make sure to:

- Follow the project's coding conventions.
- Write unit tests for new features or bug fixes.
- Update documentation as needed.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The **Galaxy Zoo** project is part of the [Zooniverse](https://www.zooniverse.org/) platform.
- Special thanks to the astronomers, researchers, and volunteers who contribute to the project.
- Data provided by the **Sloan Digital Sky Survey (SDSS)**.
- We used [TensorFlow](https://www.tensorflow.org/) for the CNN implementation.

---
