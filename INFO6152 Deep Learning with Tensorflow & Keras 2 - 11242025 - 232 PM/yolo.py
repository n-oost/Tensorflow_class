from flask import Flask, Response
import cv2
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)

# Load YOLO model (using YOLOv8 nano for speed)
model = YOLO('yolov8n.pt')  # You can download other models if needed

# Global variables for frame sharing
frame = None
lock = threading.Lock()

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam (0 is default camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, img = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model(img, stream=True)

        # Draw detections on the frame
        for r in results:
            annotated_frame = r.plot()  # Annotate with boxes and labels

        # Update global frame with lock
        with lock:
            frame = annotated_frame.copy()

        # Yield the frame in byte format for Flask response
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Object Detection with Webcam</title>
    </head>
    <body>
        <h1>Live Object Detection</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    """

if __name__ == '__main__':
    # Start frame generation in a background thread
    t = threading.Thread(target=generate_frames)
    t.daemon = True
    t.start()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)