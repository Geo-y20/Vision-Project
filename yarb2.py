from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import pytesseract
from pytesseract import Output
import numpy as np
import os
import face_recognition
from gtts import gTTS
import pygame
import io
import time

# Ensure Tesseract executable is in the PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path as necessary

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATA_FOLDER'] = 'data'

# Initialize Pygame for TTS
pygame.mixer.init()

# Load YOLOS model and image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error during text-to-speech: {e}")

def stop_audio():
    pygame.mixer.music.stop()

def detect_objects(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x1, y1, x2, y2 = map(int, box)
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        detected_objects.append(label_text)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame, detected_objects

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/object_detection')
def object_detection():
    return render_template('object_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    cap = cv2.VideoCapture(0)
    detected_objects = []
    try:
        if not cap.isOpened():
            print("Error: Could not open video device.")
            return render_template('capture_result.html', image_path='', objects='')

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return render_template('capture_result.html', image_path='', objects='')

        print("Frame captured successfully.")
        print(f"Frame shape: {frame.shape}")

        raw_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_captured_frame.png')
        cv2.imwrite(raw_output_path, frame)
        print(f"Raw captured frame saved as {raw_output_path}")

        if frame is not None and frame.size > 0:
            frame, detected_objects = detect_objects(frame)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_frame.png')
            cv2.imwrite(output_path, frame)
            print(f"Processed captured frame saved as {output_path}")
        else:
            print("Error: Captured frame is invalid.")
            return render_template('capture_result.html', image_path='', objects='')
    except Exception as e:
        print(f"Error during frame capture: {e}")
    finally:
        cap.release()

    if detected_objects:
        text_to_speech(', '.join(detected_objects))
    return render_template('capture_result.html', image_path='uploads/captured_frame.png', objects=', '.join(detected_objects))

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            text = perform_ocr(file_path)
            return render_template('ocr_result.html', text=text, image=file.filename)
    return render_template('ocr.html')

@app.route('/play_audio', methods=['POST'])
def play_audio():
    text = request.form.get('text')
    text_to_speech(text)
    return '', 204

@app.route('/stop_audio', methods=['POST'])
def stop_audio_route():
    stop_audio()
    return '', 204

def perform_ocr(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return None

    img = Image.open(image_path)
    img = np.array(img)

    try:
        text = pytesseract.image_to_string(img)
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        for i in range(len(d['level'])):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            if int(d['conf'][i]) > 70:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img = cv2.putText(img, d['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        output_image_path = os.path.splitext(image_path)[0] + '_output.png'
        cv2.imwrite(output_image_path, img)

        return text

    except pytesseract.pytesseract.TesseractNotFoundError as e:
        print(f"Tesseract not found: {e}")
        return None

@app.route('/face_recognition_page')
def face_recognition_page():
    return render_template('face_recognition.html')

@app.route('/collect_face_data', methods=['GET', 'POST'])
def collect_face_data():
    if request.method == 'POST':
        name = request.form.get('name')
        dataset_size = int(request.form.get('dataset_size'))
        data_dir = os.path.join(app.config['DATA_FOLDER'], name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        counter = 0

        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                continue

            file_path = os.path.join(data_dir, f'{counter}.jpg')
            cv2.imwrite(file_path, frame)
            print(f"Saved {file_path}")
            counter += 1
            time.sleep(0.5)

        cap.release()
        cv2.destroyAllWindows()
        
        return redirect(url_for('face_recognition_page'))
    return render_template('collect_face_data.html')

@app.route('/recognize_faces', methods=['GET'])
def recognize_faces():
    known_encodings, known_labels = load_known_faces(app.config['DATA_FOLDER'])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        labels = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            label = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                label = known_labels[match_index]

            labels.append(label)

        for (top, right, bottom, left), label in zip(face_locations, labels):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if labels:
            text_to_speech(', '.join(labels))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/face_recognition_feed')
def face_recognition_feed():
    return Response(recognize_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

def load_known_faces(data_folder):
    known_encodings = []
    known_labels = []

    for name in os.listdir(data_folder):
        person_dir = os.path.join(data_folder, name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_labels.append(name)

    return known_encodings, known_labels

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['DATA_FOLDER']):
        os.makedirs(app.config['DATA_FOLDER'])
    app.run(debug=True)
