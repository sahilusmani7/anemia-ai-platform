import os
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, classification_report
from src.exception import CustomException
from src.logger import logging


class ModelTrainer:

    def create_model(self):
        """Creates CNN model for feature extraction"""
        try:
            logging.info("Inside create_model method")

            inputs = Input(shape=(64, 64, 3))

            x1 = Conv2D(32, (2, 2), padding="same", activation="relu")(inputs)
            x2 = MaxPool2D(2, 2)(x1)

            x3 = Conv2D(64, (2, 2), padding="same", activation="relu")(x2)
            x4 = MaxPool2D(2, 2)(x3)

            x5 = Conv2D(128, (2, 2), padding="same", activation="relu")(x4)
            x6 = MaxPool2D(2, 2)(x5)

            x7 = GlobalAveragePooling2D()(x6)
            x8 = Dense(100, activation="relu")(x7)
            x9 = Dense(2, activation="sigmoid")(x8)

            model = Model(inputs=inputs, outputs=x9)

            logging.info("CNN model created successfully")
            return model

        except Exception as e:
            raise CustomException(e, sys)

    def create_feature_extractor(self, model):
        """Creates feature extractor from CNN"""
        try:
            logging.info("Creating feature extractor model")
            feature_extractor = Model(inputs=model.input, outputs=model.layers[5].output)
            return feature_extractor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_img, train_lab, test_img, test_lab):
        try:
            logging.info("Inside initiate_model_trainer method")

            os.makedirs("model", exist_ok=True)

            # Step 1: Train CNN
            model = self.create_model()
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            logging.info("Training CNN model")
            model.fit(train_img, train_lab, validation_data=(test_img, test_lab), epochs=40, batch_size=8)

            # Save CNN weights
            logging.info("Saving CNN weights")
            model.save_weights("model/model_weights.weights.h5")

            # Step 2: Create feature extractor
            feature_extractor = self.create_feature_extractor(model)

            logging.info("Extracting CNN features for Random Forest")
            train_features = feature_extractor.predict(train_img)
            test_features = feature_extractor.predict(test_img)

            train_reshaped = train_features.reshape(train_features.shape[0], -1)
            test_reshaped = test_features.reshape(test_features.shape[0], -1)

            # Step 3: Train Random Forest
            logging.info("Training Random Forest on CNN features")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(train_reshaped, train_lab)

            # Step 4: Evaluate RF
            logging.info("Evaluating Random Forest")
            y_pred_rf = rf.predict(test_reshaped)

            print("CONFUSION MATRIX")
            print(confusion_matrix(test_lab, y_pred_rf))

            print("CLASSIFICATION REPORT")
            print(classification_report(test_lab, y_pred_rf))

            # Step 5: Save RF model
            logging.info("Saving Random Forest model")
            with open("model/random_forest.pkl", "wb") as f:
                pickle.dump(rf, f)

            print("\n✅ CNN + Random Forest model training complete.")
            print("Saved:")
            print("- model/model_weights.weights.h5")
            print("- model/random_forest.pkl")

        except Exception as e:
            raise CustomException(e, sys)