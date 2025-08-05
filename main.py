import os
import cv2
import re
import shutil
import csv
from modules.detector import LicensePlateDetector
from config import INPUT_DIR, EXTRACTED_DIR, PROCESSED_DIR, DETECTED_DIR
import easyocr

# ---------- Clear Output Folders -----------
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

folders_to_clear = [DETECTED_DIR, EXTRACTED_DIR, PROCESSED_DIR]
for folder in folders_to_clear:
    clear_folder(folder)

# -------------- OCR Logic ------------------
class EasyOCRReader:
    def __init__(self, lang_list=['en'], gpu=False):
        self.reader = easyocr.Reader(lang_list, gpu=gpu)
        print(f"✅ EasyOCR initialized for languages: {lang_list} (GPU: {gpu})")

    def clean_text(self, text):
        out = re.sub(r'[^A-Z0-9]', '', text.upper())
        substitutions = {'O': '0', 'I': '1', 'S': '5', 'Q': '0'}
        out = out[:2] + ''.join(substitutions.get(c, c) for c in out[2:])
        return out

    def extract_text(self, image):
        result = self.reader.readtext(image)
        if result:
            texts = [item[1] for item in result]
            fulltext = ' '.join(texts)
            cleaned = self.clean_text(fulltext)
            print(f"✅ EasyOCR extracted: '{fulltext}' → '{cleaned}'")
            return cleaned
        else:
            print("❌ No text detected")
            return ""

# --------- Preprocessing Function ----------
def otsu_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    min_height, min_width = 40, 200
    scale_factor = max(min_width / w, min_height / h) if (w < min_width or h < min_height) else 1.0
    if scale_factor > 1.0:
        gray = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_otsu

# -------------- Pipeline Runner ---------------
if __name__ == "__main__":
    print("=== License Plate Full Recognition Pipeline (Detection + Otsu + EasyOCR) ===\n")
    detector = LicensePlateDetector()
    ocr = EasyOCRReader(['en'])

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not image_files:
        print(f"No input images found in {INPUT_DIR}")
        exit()

    all_results = []
    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        print(f"\n📸 Processing image: {img_name}")

        results = detector.detect(img_path)
        # Save visualization of detections before cropping plates!
        detector.save_detection_visualization(img_path, results)

        plates = detector.extract_plate_regions(img_path, results)
        print(f"  ➔ Plates detected: {len(plates)}")

        for i, plate_data in enumerate(plates):
            plate_img = plate_data['image']
            crop_name = f"{os.path.splitext(img_name)[0]}_plate_{i + 1}.jpg"
            # Save extracted crop
            extract_path = os.path.join(EXTRACTED_DIR, crop_name)
            cv2.imwrite(extract_path, plate_img)
            # Preprocess and save processed image
            processed = otsu_preprocess(plate_img)
            process_path = os.path.join(PROCESSED_DIR, crop_name)
            cv2.imwrite(process_path, processed)
            # OCR
            plate_number = ocr.extract_text(processed)
            print(f"    Plate {i + 1}: {plate_number}")
            all_results.append({
                "image": img_name,
                "plate_crop": crop_name,
                "recognized": plate_number
            })

    # Save results as CSV log file
    with open("results.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Input Image", "Cropped Plate", "Recognized Plate Number"])
        for r in all_results:
            writer.writerow([r["image"], r["plate_crop"], r["recognized"]])
    print("\n✅ Pipeline complete. See 'results.csv' and all output folders for results.")
