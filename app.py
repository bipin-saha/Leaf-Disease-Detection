from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize Flask application
app = Flask(__name__)

# Define directories for uploads and processed images
UPLOAD_FOLDER = os.path.join("static", "uploads")
PROCESSED_FOLDER = os.path.join("static", "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the YOLO model
model = YOLO("models/yolo_v12.pt")  # Ensure the correct model path

@app.route("/")
def index():
    """
    Renders the index page.
    """
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles image uploads, performs object detection using YOLO,
    and returns the processed image with bounding boxes.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Perform object detection
    results = model.predict(file_path)
    
    # Load the original image
    image = cv2.imread(file_path)

    # Process detection results
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
            label = model.names[int(cls)]  # Get class label
            score = conf.item()  # Confidence score

            print(f"Detected {label} with confidence {score:.2f}")

            # Draw bounding box (Red)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Display label and confidence (Blue text)
            label_text = f"{label} ({score:.2f})"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save processed image
    processed_path = os.path.join(PROCESSED_FOLDER, file.filename)
    cv2.imwrite(processed_path, image)

    return jsonify({"processed_image": f"/{processed_path}"})  

if __name__ == "__main__":
    # Run the Flask app with debugging enabled on port 5017
    app.run(debug=True, host="0.0.0.0", port=5017)
