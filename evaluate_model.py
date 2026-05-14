import os
import numpy as np
import cv2 as cv
from sklearn.metrics import confusion_matrix, classification_report
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation

# ---- IMPORTANT ----
# Try this order first (alphabetical / LabelEncoder style)
class_names = ["Resized Anemia", "Resized Non Anemia"]

# Load model
model = ModelTrainer().create_model()
model.load_weights("model/model_weights.weights.h5")

y_true = []
y_pred = []

for actual_label_index, folder_name in enumerate(class_names):
    folder_path = os.path.join("artifacts", folder_name)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        image = cv.imread(img_path)
        if image is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        # Preprocess exactly like prediction
        image = DataTransformation().image_data_transformation(image)
        image = cv.resize(image, (64, 64))
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image, verbose=0)
        predicted_label_index = np.argmax(prediction)

        y_true.append(actual_label_index)
        y_pred.append(predicted_label_index)

        print(f"{img_name} -> Predicted: {class_names[predicted_label_index]} | Actual: {folder_name}")

# Accuracy
correct = np.sum(np.array(y_true) == np.array(y_pred))
total = len(y_true)
accuracy = correct / total * 100

print("\n" + "="*50)
print(f"Total Images: {total}")
print(f"Correct Predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_true, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("="*50)