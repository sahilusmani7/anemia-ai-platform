import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.makedirs("static", exist_ok=True)

from flask import Flask, render_template, request
import cv2 as cv
import numpy as np
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation

app = Flask(__name__)

# Load model once when app starts
model = ModelTrainer().create_model()
model.load_weights("model/model_weights.weights.h5")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file and file.filename != "":
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            image = cv.imread(filepath)

            if image is not None:
                image = DataTransformation().image_data_transformation(image)
                image = cv.resize(image, (64, 64))
                image = np.expand_dims(image, axis=0)

                prediction = model.predict(image, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                # Correct class mapping
                if predicted_class == 0:
                    result = f"Anemic ({confidence:.2f}% confidence)"
                else:
                    result = f"Non-Anemic ({confidence:.2f}% confidence)"

                image_path = filepath
            else:
                result = "Could not process the uploaded image."

    return render_template("index.html", result=result, image_path=image_path)
@app.route("/health")
def health():
    return {"status": "healthy"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)