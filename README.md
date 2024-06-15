# Handwritten Digit Recognition using Machine Learning


Project Overview
This project focuses on handwritten digit recognition using a Convolutional Neural Network (CNN) and the Fashion MNIST dataset from Kaggle. The dataset contains 70,000 images of fashion items, which are used to train, test, and evaluate the performance of the CNN model.

Dataset
Source: Kaggle
Total Images: 70,000
Classes: 10 (e.g., T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)


Split:
Training Set: 60,000 images
Testing Set: 10,000 images


Model Architecture
The model is a Convolutional Neural Network (CNN) with the following architecture:

Convolutional Layer: Applies convolution operation to extract features from the input image.
Pooling Layer: Reduces the spatial dimensions of the feature maps.
Second Convolutional Layer: Further extracts features from the pooled feature maps.
Flattening: Converts the 2D matrices into a 1D vector.
Full Connection: Fully connected layer to combine features.
Output Layer: Provides the final classification output.


Implementation Steps
Data Preprocessing:


Normalize the pixel values to range [0, 1].
Reshape the images to fit the model input requirements.


Model Building:
Define the CNN architecture using layers as described above.


Model Compilation:
Compile the model with appropriate loss function, optimizer, and evaluation metric.


Model Training:
Train the model using the training dataset.
Use validation data to monitor performance and prevent overfitting.


Model Evaluation:
Evaluate the model's performance using the testing dataset.


Prediction:
Make predictions on new, unseen data.


Results
The model achieved an accuracy of XX% on the testing dataset, demonstrating its effectiveness in recognizing handwritten digits.
