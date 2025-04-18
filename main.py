from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, session, flash
from flask_session import Session  # Import Flask-Session
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mysql.connector
import hashlib

app = Flask(__name__)

# MySQL Database Configuration
db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "FER_System"
}

# Establish connection
try:
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    print("✅ Successfully connected to MySQL Database!")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(60) NOT NULL,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        predicted_emotion VARCHAR(30) NOT NULL,
        source_type ENUM('Image', 'Video') NOT NULL
    )
    """)
    db.commit()
    print("✅ Table check complete!")
except mysql.connector.Error as err:
    print(f"❌ Error: {err}")
    db = None  # Prevent app from crashing if DB connection fails

# Load trained model
MODEL_PATH = os.path.join("model", "new_model.h5")
model = load_model(MODEL_PATH, compile=False)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

app.secret_key = "your_secret_key"  # Needed for session management

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        return render_template("login.html")  # Show the login page

    data = request.get_json() or request.form  # Handle both JSON and form data
    print("📥 Received Login Data:", data)  # Debugging log

    username = data.get("username")
    password = hashlib.sha256(data.get("password").encode()).hexdigest()  # Hash password

    if not username or not password:
        return jsonify({"error": "Username and password are required!"}), 400

    cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
    user = cursor.fetchone()

    if user:
        session["user"] = username
        return jsonify({"message": "Login successful!"}), 200
    else:
        return jsonify({"error": "Invalid username or password!"}), 400

@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "GET":
        return render_template("register.html")
    
    data = request.get_json() or request.form
    username = data.get("username")
    password = data.get("password")
    confirm_password = data.get("confirm_password")

    if not username or not password or not confirm_password:
        return jsonify({"error": "All fields are required!"}), 400

    if password != confirm_password:
        return jsonify({"error": "Passwords do not match!"}), 400

    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
    db.commit()
    return jsonify({"message": "Registration successful!"}), 200

@app.route("/dashboard")
def dashboard_page():
    if 'user' not in session:
        return redirect(url_for('login_page'))

    username = session['user']  # Assuming username is stored in session after login

    # Connect to MySQL and fetch user-specific predictions
    cursor = db.cursor()
    cursor.execute("SELECT username, prediction_time, predicted_emotion, source_type FROM predictions WHERE username = %s ORDER BY prediction_time DESC", (username,))
    predictions = cursor.fetchall()
    cursor.close()

    return render_template('dashboard.html', username=username, predictions=predictions)

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/image")
def image_upload_page():
    return render_template("image.html")

@app.route("/video")
def video_page():
    return render_template("video.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
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
    
    # Save prediction to MySQL
    if "user" in session:
        username = session["user"]
        try:
            cursor.execute("INSERT INTO Predictions (username, predicted_emotion, source_type) VALUES (%s, %s, %s)",
                           (username, emotion, "Image"))
            db.commit()
            print("✅ Image prediction saved to database!")
        except mysql.connector.Error as err:
            print(f"❌ Database error: {err}")
            return jsonify({"error": "Database error!"}), 500
    
    return jsonify({"message": "File uploaded successfully!", "emotion": emotion}), 200

# Video Streaming
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
            
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
