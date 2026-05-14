# Anemia Detection using Conjunctiva Images

## Project Overview
This project is a deep learning-based image classification system that predicts whether a person is **Anemic** or **Non-Anemic** using **conjunctiva eye images**.

The system is designed as a **non-invasive screening support tool** and demonstrates how **computer vision** and **convolutional neural networks (CNNs)** can be applied in a healthcare-related classification task.

It includes:
- a trained CNN model,
- a prediction pipeline for terminal-based testing,
- a Flask web application for image upload and prediction,
- and evaluation scripts to validate model performance.

---

## 🎯 Objective
To build an image-based classification system capable of identifying whether a person is likely to be:

- **Anemic**
- **Non-Anemic**

using conjunctiva eye images as input.

---

## 🧠 Methodology
The project follows a standard machine learning / deep learning workflow:

1. **Data Collection**
   - Conjunctiva images were collected and categorized into:
     - Anemic
     - Non-Anemic

2. **Data Preprocessing**
   - Images were normalized
   - Resized to **64 × 64**
   - Labels were encoded for classification

3. **Model Building**
   - A **Convolutional Neural Network (CNN)** was developed for binary classification

4. **Model Training**
   - The CNN was trained on the processed image dataset

5. **Prediction**
   - New images can be tested using:
     - terminal-based prediction
     - Flask web interface

6. **Evaluation**
   - The trained model was validated using classification metrics and confusion matrix analysis

---

## 🏗️ Model Architecture
The project uses a custom **CNN (Convolutional Neural Network)** with the following structure:

- **Conv2D (32 filters)**
- **MaxPooling2D**
- **Conv2D (64 filters)**
- **MaxPooling2D**
- **Conv2D (128 filters)**
- **MaxPooling2D**
- **GlobalAveragePooling2D**
- **Dense Layer (100 neurons)**
- **Output Layer (2 classes)**

### Input Shape
- **64 × 64 × 3**

### Output Classes
- **Class 0 → Anemic**
- **Class 1 → Non-Anemic**

---

## 📂 Project Structure

```bash
Anemia Scope/
│
├── artifacts/                  # Dataset folders
│   ├── Resized Anemia
│   └── Resized Non Anemia
│
├── assets/                     # Sample/test images and encoder
├── model/                      # Trained model weights
│   └── model_weights.weights.h5
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   └── logger.py
│
├── static/                     # Uploaded images for web app
├── templates/
│   └── index.html              # Flask frontend UI
│
├── app.py                      # Flask web application
├── evaluate_model.py           # Model evaluation script
├── evaluate_model_advanced.py  # Advanced evaluation with saved reports
└── README.md