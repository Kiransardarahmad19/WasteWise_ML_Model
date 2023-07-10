from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

# Create Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('path_to_your_trained_model.h5')

# Define the class labels
class_labels = ['class1', 'class2']

# Define route to serve the frontend HTML file
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle image classification request
@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image file from the request
    image_file = request.files['imageFile']
    
    # Load the image using PIL
    image = Image.open(image_file)
    
    # Preprocess the image
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Perform the prediction
    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    class_label = class_labels[class_index]
    
    # Return the predicted class label
    return class_label

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
