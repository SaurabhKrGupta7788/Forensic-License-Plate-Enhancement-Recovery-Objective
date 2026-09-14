import cv2
import numpy as np
import os
import sys
import torch
from ultralytics import YOLO
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import re

from NAFNet_arch import NAFNet

try:
    import os
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: paddleocr not found. OCR extraction will be skipped.")

class NAFNetDeblur:
    def __init__(self, weights_path, device=None, tile=256, tile_overlap=32):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.net = NAFNet(
            img_channel=3,
            width=64,
            middle_blk_num=1,
            enc_blk_nums=[1,1,1,28],
            dec_blk_nums=[1,1,1,1],
        )
        checkpoint = torch.load(weights_path, map_location=device)
        state = checkpoint.get("params_ema",
                checkpoint.get("params",
                checkpoint.get("state_dict",
                checkpoint)))
        self.net.load_state_dict(state)
        self.net.to(device).eval()
        self.tile = tile
        self.tile_overlap = tile_overlap

    def _forward_tile(self, img):
        b, c, h, w = img.shape
        tile = self.tile
        overlap = self.tile_overlap
        stride = tile - overlap
        h_idx = list(range(0, h-tile, stride)) + [h-tile]
        w_idx = list(range(0, w-tile, stride)) + [w-tile]
        output = torch.zeros_like(img)
        weight = torch.zeros_like(img)
        for hi in h_idx:
            for wi in w_idx:
                patch = img[:, :, hi:hi+tile, wi:wi+tile]
                out_patch = self.net(patch)
                output[:, :, hi:hi+tile, wi:wi+tile] += out_patch
                weight[:, :, hi:hi+tile, wi:wi+tile] += 1
        return output / weight

    def __call__(self, img_rgb):
        img = torch.from_numpy(img_rgb).float()/255.0
        img = img.permute(2,0,1).unsqueeze(0).to(self.device)
        h, w = img.shape[2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if self.tile is not None:
            if h + pad_h < self.tile:
                pad_h = self.tile - h
            if w + pad_w < self.tile:
                pad_w = self.tile - w
                
        if pad_h or pad_w:
            # Use 'replicate' instead of 'constant' to prevent massive artificial black borders
            # which cause severe ringing artifacts when deblurring small plates.
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
            
        with torch.no_grad():
            if self.tile is None:
                out = self.net(img)
            else:
                out = self._forward_tile(img)
                
        out = out[:, :, :h, :w]
        out = torch.nan_to_num(out, nan=0.0)
        out = torch.clamp(out,0,1)
        out = out.squeeze(0).permute(1,2,0).cpu().numpy()
        out = (out*255).astype(np.uint8)
        return out


class LicensePlateEnhancer:
    def __init__(self, car_model_path, plate_model_path, deblur_model_path):
        self.car_model = YOLO(car_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.deblur_model = NAFNetDeblur(weights_path=deblur_model_path)
        
        if OCR_AVAILABLE:
            self.reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        else:
            self.reader = None
        
        self.VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        
    def detect_cars(self, image):
        results = self.car_model(image, verbose=False)
        cars = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls.cpu().numpy()[0])
                if cls in self.VEHICLE_CLASSES:
                    b = box.xyxy[0].cpu().numpy()
                    cars.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
        return cars

    def detect_plate(self, image):
        results = self.plate_model(image, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy()
                boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
        return boxes

    def deblur_plate(self, plate_img):
        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        deblurred_rgb = self.deblur_model(plate_rgb)
        deblurred_bgr = cv2.cvtColor(deblurred_rgb, cv2.COLOR_RGB2BGR)
        return deblurred_bgr

    def enhance_plate(self, plate_img):
        # Extremely mild sharpening so we don't destroy the NAFNet output.
        # PaddleOCR prefers natural-looking text over heavily processed binary/grayscale text.
        blur = cv2.GaussianBlur(plate_img, (3, 3), 0)
        sharpened = cv2.addWeighted(plate_img, 1.3, blur, -0.3, 0)
        return sharpened

    def evaluate_enhancement(self, original, enhanced):
        h, w = enhanced.shape[:2]
        orig_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_CUBIC)
        orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)
        
        # If enhanced is 3-channel, convert to grayscale for metric comparison
        if len(enhanced.shape) == 3 and enhanced.shape[2] == 3:
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            enhanced_gray = enhanced
            
        p = psnr(orig_gray, enhanced_gray, data_range=255)
        s = ssim(orig_gray, enhanced_gray, data_range=255)
        return p, s

    def run_ocr(self, plate_img):
        if not OCR_AVAILABLE or self.reader is None:
            return "OCR_UNAVAILABLE", 0.0
            
        try:
            results = self.reader.ocr(plate_img, cls=True)
            if not results or not results[0]:
                return "", 0.0
                
            text = ""
            confidences = []
            for line in results[0]:
                t, conf = line[1]
                text += t
                confidences.append(conf * 100.0)
                    
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            # Remove non-alphanumeric chars
            text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
            return text, avg_conf
        except Exception as e:
            return "OCR_FAILED", 0.0

    def process_image(self, image_path, output_dir):
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        filename = os.path.basename(image_path)
        cars = self.detect_cars(image)
        
        results = []
        plate_idx = 0
        
        for car_box in cars:
            cx1, cy1, cx2, cy2 = car_box
            car_crop = image[cy1:cy2, cx1:cx2]
            if car_crop.size == 0:
                continue
                
            plate_boxes = self.detect_plate(car_crop)
            
            for pbox in plate_boxes:
                px1, py1, px2, py2 = pbox
                plate_crop = car_crop[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue
                
                # Full bounding box in original image coordinates
                abs_x1 = cx1 + px1
                abs_y1 = cy1 + py1
                abs_x2 = cx1 + px2
                abs_y2 = cy1 + py2
                
                # 1. Upscale BEFORE NAFNet. 
                # This gives the deblurring network significantly more structure to work with
                # and prevents small plates from being destroyed by artificial padding edges.
                h, w = plate_crop.shape[:2]
                upscaled_crop = cv2.resize(plate_crop, (w * 3, h * 3), interpolation=cv2.INTER_LANCZOS4)
                
                # 2. Deblur the upscaled image
                deblurred_plate = self.deblur_plate(upscaled_crop)
                
                # 3. Forensic Enhance (Very mild sharpen)
                enhanced_plate = self.enhance_plate(deblurred_plate)
                
                # Evaluate (Compare original against the enhanced)
                p, s = self.evaluate_enhancement(plate_crop, enhanced_plate)
                
                # 3. OCR
                text, conf = self.run_ocr(enhanced_plate)
                
                out_crop_path = os.path.join(output_dir, f"{filename}_plate_{plate_idx}_original.jpg")
                out_enhanced_path = os.path.join(output_dir, f"{filename}_plate_{plate_idx}_enhanced.jpg")
                cv2.imwrite(out_crop_path, plate_crop)
                cv2.imwrite(out_enhanced_path, enhanced_plate)
                
                cv2.rectangle(image, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                cv2.putText(image, f"{text} ({conf:.1f}%)", (abs_x1, max(abs_y1 - 10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                results.append({
                    'plate_id': plate_idx,
                    'text': text,
                    'confidence': conf,
                    'psnr': p,
                    'ssim': s,
                    'original_crop': out_crop_path,
                    'enhanced_crop': out_enhanced_path
                })
                plate_idx += 1
            
        out_image_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(out_image_path, image)
        
        return results, out_image_path
