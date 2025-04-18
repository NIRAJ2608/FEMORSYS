from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, session
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained model
MODEL_PATH = os.path.join("model", "new_model.h5")
model = load_model(MODEL_PATH,compile=False)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
users = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET","POST"])
def register_page():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    if username in users:
        return jsonify({"error": "User already exists!"}), 400
    
    users[username] = password
    return jsonify({"message": "Registration successful!"}), 200

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        data = request.json
        username = data.get("username")
        password = data.get("password")

        if username in users and users[username] == password:
            session["user"] = username
            return jsonify({"message": "Login successful!"})
        else:
            return jsonify({"message": "Invalid credentials!"})

    return render_template("login.html")

@app.route("/dashboard")
def dashboard_page():
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template("dashboard.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/image_upload")
def image_upload_page():
    return render_template("image_upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part!"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file!"}), 400
    
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0) / 255.0
    
    prediction = model.predict(image)
    emotion = emotion_labels[np.argmax(prediction)]
    
    return jsonify({"message": "File uploaded successfully!", "emotion": emotion}), 200

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = np.expand_dims(face, axis=-1)
                face = np.expand_dims(face, axis=0) / 255.0
                prediction = model.predict(face)
                emotion = emotion_labels[np.argmax(prediction)]
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)