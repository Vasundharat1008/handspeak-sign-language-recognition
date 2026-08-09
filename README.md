# Handspeak – Sign Language Recognition

Machine learning-based gesture recognition system using flex sensors and Random Forest classification.

## 📌 Project Overview

Handspeak is a prototype sign language recognition system designed to recognize hand gestures using flex sensor readings.

The system uses five flex sensors to capture finger movements. These sensor readings are provided as input to a Random Forest machine learning model, which classifies the gesture into a predefined label.

## 🎯 Problem Statement

Communication can be challenging for people who use sign language when interacting with people who do not understand it.

Handspeak explores a technology-based solution by converting hand gestures captured through flex sensors into recognizable gesture labels.

## ⚙️ How It Works

Flex Sensors
↓
Sensor Readings
↓
Data Collection
↓
Machine Learning Model
↓
Random Forest Classifier
↓
Gesture Prediction

## 🧠 Machine Learning

The project uses a Random Forest Classifier.

### Input Features

- flex1
- flex2
- flex3
- flex4
- flex5

### Output

The model predicts the gesture label associated with the sensor readings.

## 📊 Model Training

The dataset is divided into training and testing sets using a 70/30 split.

The model uses:

- Random Forest Classifier
- 100 estimators
- Stratified train-test split
- Random state: 42

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- Feature Importance

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Machine Learning
- Flex Sensors
- Random Forest
- 
- ## 📊 Model Performance

The Random Forest classifier achieved **100% accuracy on the current calibrated test dataset**.

The model was evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- Feature Importance

> Note: The current dataset is a small calibrated prototype dataset. Therefore, the 100% test accuracy should not be interpreted as proof of real-world sign-language recognition performance. A larger and more diverse dataset would be required for robust evaluation.

## 📁 Repository Structure

```text
handspeak-sign-language-recognition/
│
├── gesture_model.py
├── gesture_data.csv
├── requirements.txt
├── models/
└── results/
