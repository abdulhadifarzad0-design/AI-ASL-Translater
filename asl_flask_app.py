print ("Loading Modules ... ")
import os, warnings, absl.logging
# === suppress warnings and tensorflow logs ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
warnings.filterwarnings("ignore")
absl.logging.set_verbosity(absl.logging.ERROR)


import cv2, uuid, shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
from gtts import gTTS
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'static/uploads'
TEMP_SOUND_DIR = 'static/tempsounds'
MODEL_PATH = 'asl_transfer_model.keras' 
IMG_SIZE = 128
print("Using Model:", MODEL_PATH)
model = load_model(MODEL_PATH)
class_names = np.load("classes.npy")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def recreate_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)  
    os.makedirs(path)         

recreate_directory(UPLOAD_FOLDER)
recreate_directory(TEMP_SOUND_DIR)

print("Starting Server...\n\n")

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = float(preds[class_id])
    return class_names[class_id], confidence

def speak(text):
    filename = f"sound_{uuid.uuid4().hex}.mp3"
    path = os.path.join(TEMP_SOUND_DIR, filename)
    gTTS(text=text).save(path)
    return filename

@app.route('/')
def index():
    return render_template('asl_client.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'})

    file = request.files['image']
    filename = file.filename or 'webcam.jpg'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    label, confidence = predict_image(filepath)
    return jsonify({
        'label': label,
        'confidence': confidence
    })

@app.route('/speak', methods=['POST'])
def tts():
    data = request.get_json()
    label = data.get('text', '')
    filename = speak(f"The sign is {label}")
    return jsonify({'filename': filename})

@app.route('/sounds/<filename>')
def serve_sound(filename):
    return send_from_directory(TEMP_SOUND_DIR, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
