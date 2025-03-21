from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('soil_classification_model.h5')

# Define the classes for soil types
soil_classes = [
    'Alluvial Soil', 'Black Soil', 'Cinder Soil', 'Clay Soil',
    'Laterite Soil', 'Peat Soil', 'Red Soil', 'Yellow Soil'
]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join('/tmp', file.filename)
        file.save(file_path)
        processed_image = preprocess_image(file_path)
        predictions = model.predict(processed_image)
        predicted_class = soil_classes[np.argmax(predictions)]
        os.remove(file_path)
        return jsonify({'soil_type': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)