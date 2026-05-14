import cv2 as cv
import numpy as np
import sys
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
class PredictPipeline:

    def predict(self, img_path):
        """This method will predict whether the person has anemia or not"""

        try:
            logging.info("reading the image")
            image = cv.imread(img_path)

            if image is None:
                print(f"Error: Could not read image from path -> {img_path}")
                return

            logging.info("creating and loading model")
            model = ModelTrainer().create_model()
            model.load_weights('model/model_weights.weights.h5')

            logging.info("preprocessing the image")

            # Normalize
            preprocessed_image = DataTransformation().image_data_transformation(image)

            # Resize to match training input
            preprocessed_image = cv.resize(preprocessed_image, (64, 64))

            # Add batch dimension
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

            logging.info("prediction")
            prediction = model.predict(preprocessed_image)

            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            print("\nPrediction probabilities:", prediction)

            # CORRECT LABEL MAPPING
            if predicted_class == 0:
                print(f"Result: Anemic ({confidence:.2f}% confidence)")
            else:
                print(f"Result: Non-Anemic ({confidence:.2f}% confidence)")

        except Exception as e:
            print("Error during prediction:", e)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = 'assets/test_img.jpg'

    pp = PredictPipeline()
    pp.predict(img_path)