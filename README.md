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
![Demo Screenshot](link_to_screenshot.png)

## Key Features

- **Real-time Content Analysis**: Immediate detection of cyberbullying language.
- **Optimized Machine Learning Pipeline**: A mix of LinearSVC and Logistic Regression models to achieve high accuracy.
- **Deployable in Flask**: Flask-based backend allows simple deployment and scaling.

## Data Preprocessing and Engineering

- **Data Cleaning**: Removed patterns like `@user` mentions, URLs, and special characters.
- **Tokenization and Lemmatization**: Tokenized and lemmatized the text to normalize it, keeping meaningful keywords.
- **Text Vectorization**: Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical format suitable for model training.
- **Stopwords Removal**: Custom stopwords were implemented to exclude common but non-informative words.

