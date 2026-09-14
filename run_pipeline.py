import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import glob
import pandas as pd
from pipeline import LicensePlateEnhancer
import sys

def main():
    # Paths updated to reflect the full pipeline
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    car_model_path = os.path.join(BASE_DIR, "models", "yolov10m.pt")
    plate_model_path = os.path.join(BASE_DIR, "models", "plate_yolov8s.pt")
    deblur_model_path = os.path.join(BASE_DIR, "models", "nafnet.pth")
    
    input_dir = os.path.join(BASE_DIR, "test_images")
    output_dir = os.path.join(BASE_DIR, "outputs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading License Plate Enhancer Pipeline (Car YOLO -> Plate YOLO -> NAFNet Deblur -> PaddleOCR)...")
    try:
        enhancer = LicensePlateEnhancer(car_model_path, plate_model_path, deblur_model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)
        
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))[:10]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images to process.")
    
    all_results = []
    
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing {os.path.basename(img_path)}...")
        try:
            results, out_image = enhancer.process_image(img_path, output_dir)
            if results:
                for r in results:
                    r['filename'] = os.path.basename(img_path)
                    r['output_image'] = out_image
                    all_results.append(r)
            else:
                print(f"No plates found in {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['filename', 'plate_id', 'text', 'confidence', 'psnr', 'ssim', 'original_crop', 'enhanced_crop']]
        report_path = os.path.join(output_dir, "evaluation_report.csv")
        df.to_csv(report_path, index=False)
        print(f"\nProcessing complete! Report saved to {report_path}")
        print("\nSummary of results:")
        for res in all_results:
            print(f"- {res['filename']}: Text='{res['text']}', Conf={res['confidence']:.1f}%, PSNR={res['psnr']:.2f}")
    else:
        print("No plates were successfully processed.")

if __name__ == "__main__":
    main()
