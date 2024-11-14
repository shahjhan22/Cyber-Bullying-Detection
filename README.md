# Cyber-Bullying-Detection

This repository contains a Flask-based cyberbullying detection system that identifies and classifies toxic online content. The system leverages machine learning models trained to recognize harmful language and is optimized for accuracy and efficiency, ensuring a safe digital experience.

## Problem Statement

Cyberbullying is a growing concern in online communities, with individuals increasingly exposed to harmful content. This project aims to automatically detect and classify cyberbullying content from non-toxic content using machine learning. By automating this detection, we aim to help platforms, moderators, and users proactively address and reduce online harassment.

## Table of Contents

- [Project Demo](#project-demo)
- [Key Features](#key-features)
- [Data Preprocessing and Engineering](#data-preprocessing-and-Engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Selected Models](#selected-models)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Working Flow](#working-flow)
- [Conclusion](#conclusion)

## Project Demo

### Frontend UI
The project uses a Flask web application with an intuitive interface:
1. A user inputs text they want analyzed.
2. The application processes the text and displays whether the content is classified as "Bullying" or "Non-Bullying."

#### Preview

https://github.com/user-attachments/assets/0e505eea-acf9-47bf-9d24-577e157eb9fb

## Key Features

- **Real-time Content Analysis**: Immediate detection of cyberbullying language.
- **Optimized Machine Learning Pipeline**: A mix of LinearSVC and Logistic Regression models to achieve high accuracy.
- **Deployable in Flask**: Flask-based backend allows simple deployment and scaling.

## Data Preprocessing and Engineering

- **Data Cleaning**: Removed patterns like `@user` mentions, URLs, and special characters.
- **Tokenization and Lemmatization**: Tokenized and lemmatized the text to normalize it, keeping meaningful keywords.
- **Text Vectorization**: Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical format suitable for model training.
- **Stopwords Removal**: Custom stopwords were implemented to exclude common but non-informative words.

## Model Training and Evaluation

In this phase, multiple machine learning models were trained and evaluated on the processed data. The models were chosen for their ability to handle text classification tasks effectively. The following steps were followed during model training and evaluation:

1. **Training Process**: All models were trained on the TF-IDF vectorized text data, which served as input features for classification.
2. **Evaluation Metrics**: Each model was evaluated using the following metrics:
   - **Accuracy**: Measures the overall correctness of the model.
   - **Precision**: Indicates how many of the predicted positive labels are actually correct.
   - **Recall**: Measures how many actual positives were correctly identified.
   - **F1-Score**: A balance between Precision and Recall, especially useful when dealing with imbalanced datasets.
3. **Model Performance Comparison**: All models were compared to determine which gave the best performance based on the evaluation metrics.

Models trained during this phase included:
- **LinearSVC**
- **Logistic Regression**
- **Naive Bayes**


## Selected Models

After testing several machine learning algorithms, the following models were selected based on their performance metrics:

- **LinearSVC**: Chosen for its strong performance in text classification tasks and ability to handle high-dimensional sparse data efficiently.
- **Logistic Regression**: A solid baseline model that provided good results with relatively less complexity.

These models were selected for their balance between training time, prediction performance, and ability to generalize to unseen data.

## Hyperparameter Tuning

To optimize model performance, hyperparameter tuning was conducted using Grid Search and Randomized Search techniques. This process involved the following steps:

1. **Hyperparameter Selection**: The key hyperparameters for each model were identified for tuning, such as:
   - **LinearSVC**: Regularization strength (C), kernel type, and max iterations.
   - **Logistic Regression**: Penalty type (L1, L2), regularization strength (C), and solver type.
   
2. **Grid Search**: For each model, a range of possible values for hyperparameters was defined, and the model was trained with each combination to identify the best-performing one.
   
3. **Randomized Search**: As an alternative to Grid Search, Randomized Search was used for a more efficient search over hyperparameters by selecting random combinations within a specified range.


## Conclusion

This project effectively demonstrated the power of machine learning in the task of cyberbullying detection. After thorough data preprocessing, model training, and hyperparameter tuning, the selected models achieved high accuracy, precision, and recall in detecting toxic and non-toxic online content. Future improvements can include fine-tuning additional models and expanding the dataset to handle more complex forms of online toxicity.


