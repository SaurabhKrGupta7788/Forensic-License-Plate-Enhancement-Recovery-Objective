# CerebralZip Assignment 2 — Forensic License Plate Enhancement & Recovery

This repository contains a full forensic-style image and video enhancement pipeline that recovers readable license plate text from degraded footage.

## Features
- **Dual-YOLO Detection:** Detects cars first, then exact plate regions to minimize false positives.
- **Pre-Upscaling Strategy:** Upscales small plates with Lanczos4 interpolation *before* deblurring to give deep learning models maximum structural context.
- **NAFNet Deblurring:** Utilizes a custom NAFNet architecture (with edge-replicate padding) to reverse optical and motion blur.
- **Bilateral Filtering:** Smooths out noise without destroying text edges.
- **PaddleOCR Extraction:** State-of-the-art text extraction on the final enhanced 3-channel image.

## Setup Instructions

1. Ensure you have Python installed and create a virtual environment (e.g., Anaconda).
2. Install the exact requirements needed:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the `models/` directory contains:
   - `yolov10m.pt` (Car detection)
   - `plate_yolov8s.pt` (Plate detection)
   - `nafnet.pth` (Deblurring weights)

## How to Run the Application

This project provides two fully featured UI options.

### Option 1: Streamlit (Recommended)
To run the primary Streamlit interface with Web Camera and Image Upload support, run the provided batch script to bypass protobuf issues:
```bash
.\run_streamlit.bat
```
*(Or manually set `$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` and run `streamlit run app.py`)*

### Option 2: Flask
If you prefer a lightweight Flask server:
```bash
python flask_app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Generating Final Sample Outputs
To automatically process a folder of test images (like the 5 required cases) and generate before/after comparisons:
1. Place the test images in the `final_test_images/` directory.
2. Run the generation script:
   ```bash
   python generate_final_samples.py
   ```
3. The before/after plate crops and final annotated images will be saved in the `final_outputs/` directory.

## Deliverables
- **Report.md**: Detailed methodology, metric justifications (PSNR/SSIM), and failure case analysis.
- **Source Code**: Fully self-contained in this directory.
- **Sample Outputs**: Available in the `final_outputs/` directory.
