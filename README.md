# Handwritten Digit Recognition using Machine Learning

## Project Overview

This project focuses on handwritten digit recognition using a Convolutional Neural Network (CNN) and the Fashion MNIST dataset from Kaggle. The dataset contains 70,000 images of fashion items, which are used to train, test, and evaluate the performance of the CNN model.

## Dataset

- **Source:** [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)
- **Total Images:** 70,000
- **Classes:** 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

### Split

- **Training Set:** 60,000 images
- **Testing Set:** 10,000 images

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

1. **Convolutional Layer:** Applies convolution operation to extract features from the input image.
2. **Pooling Layer:** Reduces the spatial dimensions of the feature maps.
3. **Second Convolutional Layer:** Further extracts features from the pooled feature maps.
4. **Flattening:** Converts the 2D matrices into a 1D vector.
5. **Fully Connected Layer:** Combines features from all the previous layers.
6. **Output Layer:** Provides the final classification output.

## Implementation Steps

### 1. Data Preprocessing

- **Normalize:** Normalize the pixel values to the range [0, 1].
- **Reshape:** Reshape the images to fit the model input requirements (e.g., (28, 28, 1) for grayscale images).

### 2. Model Building

- **Define Architecture:** Construct the CNN model using Keras or TensorFlow, defining the layers as described in the model architecture section.

### 3. Model Compilation

- **Loss Function:** Use an appropriate loss function such as `categorical_crossentropy`.
- **Optimizer:** Choose an optimizer like `Adam`.
- **Metrics:** Evaluate the model using metrics such as `accuracy`.

### 4. Model Training

- **Train Model:** Train the model using the training dataset.
- **Validation:** Use validation data to monitor performance and prevent overfitting by employing techniques such as early stopping or dropout.

### 5. Model Evaluation

- **Evaluate Performance:** Evaluate the model's performance using the testing dataset to determine its accuracy, precision, recall, and F1 score.

### 6. Prediction

- **Make Predictions:** Use the trained model to make predictions on new, unseen data.

## Dependencies

- **Python:** 3.x
- **Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Swapnilghait/Handwritten-Digit-Recognition.git
   cd Handwritten-Digit-Recognition
