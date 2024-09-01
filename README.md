# facialemotionrecognition
This project implements a deep learning model for classifying emotions from images into seven categories: surprise, fear, angry, neutral, sad, disgust, and happy. The solution involves training a Convolutional Neural Network (CNN) to recognize emotions and deploying the model using a Streamlit web application for real-time interaction.

# Features
Deep Learning Model: A CNN architecture trained to classify emotions from images. The model includes multiple convolutional layers, max-pooling, dropout, and dense layers to ensure robust feature extraction and accurate predictions.
Image Upload and Prediction: Users can upload images through the Streamlit interface, which preprocesses the image and feeds it to the model to generate predictions.
Real-Time Feedback: Displays the predicted emotion label based on the uploaded image.

# Model Details
The CNN model architecture includes:
Multiple Conv2D and MaxPooling2D layers.
Dropout layers to prevent overfitting.
Fully connected Dense layers.
Softmax output layer for classification.
