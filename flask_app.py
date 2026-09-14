import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from pipeline import LicensePlateEnhancer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'outputs'

# Load pipeline models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
car_model = os.path.join(BASE_DIR, "models", "yolov10m.pt")
plate_model = os.path.join(BASE_DIR, "models", "plate_yolov8s.pt")
deblur_model = os.path.join(BASE_DIR, "models", "nafnet.pth")
enhancer = LicensePlateEnhancer(car_model, plate_model, deblur_model)

camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None

def generate_frames():
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        cars = enhancer.detect_cars(frame)
        for car_box in cars:
            cx1, cy1, cx2, cy2 = car_box
            car_crop = frame[cy1:cy2, cx1:cx2]
            if car_crop.size == 0: continue
            
            plate_boxes = enhancer.detect_plate(car_crop)
            for pbox in plate_boxes:
                px1, py1, px2, py2 = pbox
                plate_crop = car_crop[py1:py2, px1:px2]
                if plate_crop.size == 0: continue
                
                abs_x1, abs_y1 = cx1 + px1, cy1 + py1
                abs_x2, abs_y2 = cx1 + px2, cy1 + py2
                
                try:
                    h, w = plate_crop.shape[:2]
                    upscaled = cv2.resize(plate_crop, (w * 3, h * 3), interpolation=cv2.INTER_LANCZOS4)
                    deblurred = enhancer.deblur_plate(upscaled)
                    enhanced = enhancer.enhance_plate(deblurred)
                    text, conf = enhancer.run_ocr(enhanced)
                    
                    cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0,255,0), 2)
                    cv2.putText(frame, f"{text} ({conf:.1f}%)", (abs_x1, max(abs_y1-10, 0)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                except:
                    pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    release_camera()
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    release_camera()
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "flask_upload.jpg")
        file.save(filepath)
        results, out_path = enhancer.process_image(filepath, app.config['UPLOAD_FOLDER'])
        
        output_html = "<h2>Processing Complete</h2>"
        output_html += f"<p>Image saved to {out_path}</p>"
        if results:
            for r in results:
                output_html += f"<p><b>Plate:</b> {r['text']} (Conf: {r['confidence']:.2f}%)</p>"
        else:
            output_html += "<p>No plates detected.</p>"
            
        output_html += "<br><a href='/'>Go Back</a>"
        return output_html

if __name__ == '__main__':
    # Running with debug=False prevents Flask from spinning up a second background 
    # process, which was causing the Out Of Memory (OOM) crash by loading the models twice.
    app.run(host='0.0.0.0', port=5000, debug=False)
