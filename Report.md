# Assignment 2 — Forensic License Plate Enhancement & Recovery
## Project Report

### 1. Approach and Pipeline Architecture
The pipeline is designed to overcome common degradations in license plate recognition (e.g., blur, low resolution, noise, perspective distortion). The architecture consists of three main stages, optimized for maximum OCR accuracy on real-world datasets:

**A. Dual-Stage Detection & Localization**
- **Vehicle Detection:** Utilized a pre-trained YOLOv10m model (`yolov10m.pt`) to first detect bounding boxes around vehicles. This minimizes false positives (e.g., street signs or text on billboards).
- **Plate Detection:** The cropped vehicle region is passed into a fine-tuned YOLOv8 small model (`plate_yolov8s.pt`) to isolate the exact license plate Region of Interest (ROI). This ensures the enhancement algorithms only process relevant pixels.

**B. Upscaling & Forensic Restoration**
Rather than processing a small, low-resolution plate directly, the pipeline employs an upscale-first strategy to give deep learning models significantly more structural data to work with:
1. **Lanczos4 Upsampling:** The cropped plate is scaled up by a factor of 3 using `cv2.INTER_LANCZOS4`. Lanczos interpolation is vastly superior for text upscaling compared to bicubic/bilinear methods, as it preserves crisp edges without creating jagged pixel blocks.
2. **Deep Deblurring (NAFNet):** The upscaled image is passed through a pre-trained NAFNet (Nonlinear Activation Free Network) model to reverse motion blur and optical defocus. (Note: NAFNet was fixed to use `replicate` padding instead of `constant` black padding to prevent severe artificial edge ringing on small plate crops).
3. **Mild Post-Sharpening:** A very gentle Gaussian blur and Unsharp Mask is applied to the deblurred output. Previous iterations used aggressive NLMeans Denoising and CLAHE, but these were found to artificially break character strokes (e.g., turning a '4' into an '1' or 'L'). The new, milder approach preserves the natural structure of the text, which modern OCR engines prefer.

**C. OCR Extraction**
- **PaddleOCR:** Utilized `paddleocr`, a state-of-the-art deep-learning based text recognition network. The pipeline passes the 3-channel RGB enhanced plate directly to PaddleOCR, which internally handles binarization and alignment. It outperforms Tesseract significantly on noisy, varied fonts.

### 2. Evaluation Metrics
To quantitatively evaluate the success of the enhancement, two image quality metrics were used alongside OCR confidence:
- **PSNR (Peak Signal-to-Noise Ratio):** Measures the absolute pixel difference between the resized original crop and the enhanced crop.
- **SSIM (Structural Similarity Index):** Measures structural changes. A moderate SSIM indicates that while blur and noise were removed, the fundamental structural shapes (characters) were preserved.
- **OCR Confidence:** The primary functional metric. The pipeline extracts the average confidence score from PaddleOCR.

### 3. Failure Cases and Limitations
While the pipeline effectively handles low resolution and moderate blur, certain degradations remain challenging:
- **Extreme Perspective Distortion:** Plates viewed from severe angles (e.g., > 60 degrees) cause characters to overlap. Homography transformations require detecting the 4 exact corners of the plate, which is unstable with standard rectangular YOLO bounding boxes.
- **Total Saturation:** When license plates are completely blown out (white pixels = 255) due to direct flash or headlights, the original signal is unrecoverable, making restoration impossible.
- **Zero-Information Motion Blur:** If the motion blur kernel is larger than the stroke width of the characters, NAFNet struggles to hallucinate the correct character structure.

### 4. Conclusion
The pipeline successfully bridges the gap between raw, degraded YOLO detections and readable OCR output. By restructuring the pipeline to upscale the image *before* passing it into the NAFNet deblurring model, the system provides significantly higher fidelity text recovery, overcoming edge-padding artifacts and preserving critical character strokes.
