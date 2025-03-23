from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
from keras.models import model_from_json
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # Allows requests from React

# Load the model architecture and weights
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to preprocess input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from Flask!", "items": ["Item1", "Item2", "Item3"]})

@app.route('/api/recognize', methods=['POST'])
def recognize_emotion():
    try:
        # Read image from request
        data = request.json
        image_data = base64.b64decode(data['image'])

        # Convert image to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        # Convert to grayscale for prediction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        predictions = []
        for (x, y, w, h) in faces:
            face_image = gray[y:y+h, x:x+w]
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            predictions.append(prediction_label)

        if predictions:
            return jsonify({"mood": predictions[0]})
        else:
            return jsonify({"mood": "No face detected"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process image"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
