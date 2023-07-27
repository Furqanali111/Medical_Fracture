import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib


dataset_folder = "dataset/train"
val_dataset_folder = "dataset/val"

def load_images(folder_path):
    images = []
    labels = []
    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = load_img(img_path, target_size=(224, 224))  # Resizing to match VGG16 input size
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(class_name)
    return np.array(images), np.array(labels)

X, y = load_images(dataset_folder)


vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(images):
    features = vgg_model.predict(images)
    return features.reshape(features.shape[0], -1)

X_features = extract_features(X)


X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
classifier = LogisticRegression()
X_val, y_val = load_images(val_dataset_folder)
X_val_features = extract_features(X_val)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_val_pred = classifier.predict(X_val_features)

val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model_filename = "trained_model.joblib"
joblib.dump(classifier, model_filename)
print("Model saved as:", model_filename)