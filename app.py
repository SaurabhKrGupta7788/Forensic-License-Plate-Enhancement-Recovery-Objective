import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

from pipeline import LicensePlateEnhancer

# Initialize Streamlit UI
st.set_page_config(page_title="License Plate Recognition System", layout="wide")
st.title("License Plate Detection and OCR Pipeline")
st.write("This application detects cars, localizes license plates, enhances them, and performs OCR.")

@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    car_model_path = os.path.join(BASE_DIR, "models", "yolov10m.pt")
    plate_model_path = os.path.join(BASE_DIR, "models", "plate_yolov8s.pt")
    deblur_model_path = os.path.join(BASE_DIR, "models", "nafnet.pth")
    
    enhancer = LicensePlateEnhancer(
        car_model_path=car_model_path,
        plate_model_path=plate_model_path,
        deblur_model_path=deblur_model_path
    )
    return enhancer

try:
    enhancer = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Add a sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a mode", ["Upload Image", "Live Web Camera"])

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)

if app_mode == "Upload Image":
    st.header("Image Upload and Processing")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.image(image, channels="BGR", caption="Uploaded Image")
        
        if st.button("Process Image"):
            with st.spinner("Processing..."):
                # Save uploaded image to temp path
                temp_path = os.path.join(output_dir, "temp_upload.jpg")
                cv2.imwrite(temp_path, image)
                
                results, out_image_path = enhancer.process_image(temp_path, output_dir)
                
                if out_image_path and os.path.exists(out_image_path):
                    out_image = cv2.imread(out_image_path)
                    st.image(out_image, channels="BGR", caption="Processed Output")
                
                if results:
                    st.subheader("Extracted Plates:")
                    for res in results:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if os.path.exists(res['original_crop']):
                                st.image(cv2.imread(res['original_crop']), channels="BGR", caption="Original Plate Crop")
                        with col2:
                            if os.path.exists(res['enhanced_crop']):
                                st.image(cv2.imread(res['enhanced_crop']), channels="BGR", caption="Enhanced Plate")
                        with col3:
                            st.write(f"**Plate ID:** {res['plate_id']}")
                            st.write(f"**Detected Text:** {res['text']}")
                            st.write(f"**Confidence:** {res['confidence']:.2f}%")
                            st.write(f"**PSNR:** {res['psnr']:.2f}")
                            st.write(f"**SSIM:** {res['ssim']:.2f}")
                        st.markdown("---")
                else:
                    st.warning("No license plates were detected in the image.")

elif app_mode == "Live Web Camera":
    st.header("Live Web Camera Detection")
    st.write("Click 'Start' to begin live detection. Ensure your webcam is connected.")
    
    run = st.checkbox("Start Live Feed")
    
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        # Add a placeholder for live results
        results_placeholder = st.empty()
        
        # Check if camera opened successfully
        if not cap.isOpened():
            st.error("Error opening video stream or file. Make sure camera is available.")
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from camera.")
                break
                
            # To avoid saving to disk constantly in a loop, we adapt `process_image` logic directly
            cars = enhancer.detect_cars(frame)
            results = []
            plate_idx = 0
            
            for car_box in cars:
                cx1, cy1, cx2, cy2 = car_box
                car_crop = frame[cy1:cy2, cx1:cx2]
                if car_crop.size == 0:
                    continue
                    
                plate_boxes = enhancer.detect_plate(car_crop)
                
                for pbox in plate_boxes:
                    px1, py1, px2, py2 = pbox
                    plate_crop = car_crop[py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        continue
                    
                    abs_x1 = cx1 + px1
                    abs_y1 = cy1 + py1
                    abs_x2 = cx1 + px2
                    abs_y2 = cy1 + py2
                    
                    h, w = plate_crop.shape[:2]
                    upscaled = cv2.resize(plate_crop, (w * 3, h * 3), interpolation=cv2.INTER_LANCZOS4)
                    deblurred_plate = enhancer.deblur_plate(upscaled)
                    enhanced_plate = enhancer.enhance_plate(deblurred_plate)
                    text, conf = enhancer.run_ocr(enhanced_plate)
                    
                    cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{text} ({conf:.1f}%)", (abs_x1, max(abs_y1 - 10, 0)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                    results.append({
                        'text': text,
                        'confidence': conf
                    })
                    plate_idx += 1
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
            # Display latest results
            if results:
                with results_placeholder.container():
                    st.write("Latest Detections:")
                    for r in results:
                        st.write(f"- {r['text']} (Conf: {r['confidence']:.1f}%)")
            
            # Allow some time for UI update
            time.sleep(0.01)
        
        if 'cap' in locals() and cap is not None:
            cap.release()
