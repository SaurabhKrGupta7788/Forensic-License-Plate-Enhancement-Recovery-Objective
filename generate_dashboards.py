import os
import cv2
import numpy as np
from pipeline import LicensePlateEnhancer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
car_model = os.path.join(BASE_DIR, "models", "yolov10m.pt")
plate_model = os.path.join(BASE_DIR, "models", "plate_yolov8s.pt")
deblur_model = os.path.join(BASE_DIR, "models", "nafnet.pth")

print("Initializing Pipeline for Dashboards...")
enhancer = LicensePlateEnhancer(car_model, plate_model, deblur_model)

input_dir = os.path.join(BASE_DIR, "final_test_images")
output_dir = os.path.join(BASE_DIR, "final_outputs")
os.makedirs(output_dir, exist_ok=True)

def create_dashboard(orig_path, res_path, results_info, out_path):
    orig = cv2.imread(orig_path)
    res = cv2.imread(res_path)
    
    if orig is None or res is None: return
    
    # 1. Top row (Original vs Result)
    target_w = 800
    orig_h = int(orig.shape[0] * (target_w / orig.shape[1]))
    res_h = int(res.shape[0] * (target_w / res.shape[1]))
    
    orig_resized = cv2.resize(orig, (target_w, orig_h))
    res_resized = cv2.resize(res, (target_w, res_h))
    
    max_top_h = max(orig_h, res_h)
    
    # 2. Canvas setup
    margin = 30
    plate_row_h = 180
    total_w = target_w * 2 + margin * 3
    # Ensure canvas is big enough if there are no plates detected
    total_h = max_top_h + margin * 3 + (plate_row_h * max(len(results_info), 1)) + margin
    
    # Dark grey background
    canvas = np.full((total_h, total_w, 3), 30, dtype=np.uint8)
    
    # Put top images
    canvas[margin:margin+orig_h, margin:margin+target_w] = orig_resized
    canvas[margin:margin+res_h, target_w + margin*2 : target_w*2 + margin*2] = res_resized
    
    # Titles
    cv2.putText(canvas, "Original Input", (margin, margin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(canvas, "Processed Output (Detected Box)", (target_w + margin*2, margin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # 3. Plate rows
    current_y = max_top_h + margin * 2
    cv2.putText(canvas, "Extracted Plates & Enhancement Results:", (margin, current_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    current_y += 60
    if not results_info:
        cv2.putText(canvas, "No License Plates Detected.", (margin, current_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    for r in results_info:
        oc = cv2.imread(r['original_crop'])
        ec = cv2.imread(r['enhanced_crop'])
        
        if oc is not None:
            # Scale to fit max width 200 and max height 140
            scale = min(200 / oc.shape[1], 140 / oc.shape[0])
            ocw, och = int(oc.shape[1] * scale), int(oc.shape[0] * scale)
            ocr = cv2.resize(oc, (ocw, och))
            canvas[current_y : current_y+och, margin : margin+ocw] = ocr
            cv2.putText(canvas, "Original Crop", (margin, current_y + och + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
            
        if ec is not None:
            # Scale to fit max width 350 and max height 140
            scale = min(350 / ec.shape[1], 140 / ec.shape[0])
            ecw, ech = int(ec.shape[1] * scale), int(ec.shape[0] * scale)
            ecr = cv2.resize(ec, (ecw, ech))
            canvas[current_y : current_y+ech, margin + 250 : margin + 250 + ecw] = ecr
            cv2.putText(canvas, "Enhanced Crop", (margin + 250, current_y + ech + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
            
        # Text details
        text_x = margin + 650
        cv2.putText(canvas, f"Detected Text: {r['text']}", (text_x, current_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(canvas, f"Confidence: {r['confidence']:.2f}%", (text_x, current_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(canvas, f"Quality Metrics -> PSNR: {r['psnr']:.2f} | SSIM: {r['ssim']:.2f}", (text_x, current_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        current_y += plate_row_h
        
    cv2.imwrite(out_path, canvas)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    img_path = os.path.join(input_dir, filename)
    print(f"Processing {filename}...")
    
    # 1. Process image using the pipeline
    results, out_image_path = enhancer.process_image(img_path, output_dir)
    
    if out_image_path:
        # 2. Create the unified dashboard image
        dash_path = os.path.join(output_dir, f"Dashboard_{filename}.jpg")
        create_dashboard(img_path, out_image_path, results, dash_path)
        print(f"Created Dashboard: {dash_path}")

print("All dashboards generated successfully!")
