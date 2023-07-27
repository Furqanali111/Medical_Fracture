import joblib
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array


model_filename = "trained_model.joblib"
loaded_classifier = joblib.load(model_filename)

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


X = "dataset/val/not fractured/3.jpg"
X_features = preprocess_image(X)

X_features = vgg_model.predict(X_features)
X_features = X_features.reshape(X_features.shape[0], -1)

predictions = loaded_classifier.predict(X_features)

print(predictions)
