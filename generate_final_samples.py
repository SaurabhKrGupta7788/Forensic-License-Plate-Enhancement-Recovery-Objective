import os
import cv2
from pipeline import LicensePlateEnhancer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
car_model = os.path.join(BASE_DIR, "models", "yolov10m.pt")
plate_model = os.path.join(BASE_DIR, "models", "plate_yolov8s.pt")
deblur_model = os.path.join(BASE_DIR, "models", "nafnet.pth")

print("Initializing Pipeline...")
enhancer = LicensePlateEnhancer(car_model, plate_model, deblur_model)

input_dir = os.path.join(BASE_DIR, "final_test_images")
output_dir = os.path.join(BASE_DIR, "final_outputs")
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    img_path = os.path.join(input_dir, filename)
    print(f"Processing {filename}...")
    enhancer.process_image(img_path, output_dir)
    print(f"Finished {filename}. Results saved in {output_dir}")

print("All sample outputs generated successfully!")
