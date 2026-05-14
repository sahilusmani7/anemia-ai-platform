import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation

# Correct class mapping
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
            continue

        image = DataTransformation().image_data_transformation(image)
        image = cv.resize(image, (64, 64))
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image, verbose=0)
        predicted_label_index = np.argmax(prediction)

        y_true.append(actual_label_index)
        y_pred.append(predicted_label_index)

# Metrics
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)

# Save confusion matrix image
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# Save report
with open("classification_report.txt", "w") as f:
    f.write(report)

accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true) * 100

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nSaved:")
print(" - confusion_matrix.png")
print(" - classification_report.txt")