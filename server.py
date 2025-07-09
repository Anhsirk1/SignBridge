from flask import Flask, request, jsonify, send_from_directory
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import os

app = Flask(__name__)

# Initialize Roboflow API client with your API key
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",  # Roboflow API URL
    api_key="LVe5z1ouGuesEUDVwmk1"  # Your Roboflow API key
)

# Serve the index.html file for the root route
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

# Route to handle image uploads and predictions
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:  # Check if the file is sent in the request
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']  # Get the uploaded file
    try:
        img = Image.open(file.stream)  # Open image from the file stream
    except Exception as e:
        return jsonify({'error': f'Error processing image: {e}'}), 400

    # Save the image to a temporary file (optional, you can remove this if unnecessary)
    img_path = 'temp.jpg'
    img.save(img_path)

    try:
        # Use Roboflow's inference function to get predictions
        result = CLIENT.infer(img_path, model_id="indian-sign-language-detection/1")
    except Exception as e:
        return jsonify({'error': f'Error with Roboflow inference: {e}'}), 500

    # Optionally remove the temp image after inference (cleanup)
    os.remove(img_path)

    return jsonify(result)  # Return the result as JSON

if __name__ == '__main__':
    app.run(port=5000)  # Run the Flask server on port 5000