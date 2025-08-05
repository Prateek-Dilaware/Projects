import os
import cv2
from modules.detector import LicensePlateDetector
from config import INPUT_DIR, EXTRACTED_DIR

def otsu_preprocess(image):
    """Preprocess plate crop for OCR using Otsu binarization."""
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

def test_license_plate_detection():
    print("=== License Plate Detection & Preprocessing Test ===\n")

    # Ensure required folders exist
    processed_dir = "data/outputs/processed/"
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Initialize detector
    try:
        detector = LicensePlateDetector()
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False

    test_image = os.path.join(INPUT_DIR, "sample.jpg")
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Please place a test image in data/input/sample.jpg")
        return False

    print(f"📸 Testing with image: {test_image}")

    # Perform detection
    print("\n--- Running Detection ---")
    results = detector.detect(test_image)
    if results is None:
        print("❌ Detection failed")
        return False

    detection_info = detector.get_detection_info(results)
    print(f"\n--- Detection Results ---")
    print(f"🎯 Total license plates found: {detection_info['total_plates']}")

    if detection_info['total_plates'] == 0:
        print("❌ No license plates detected")
        return False

    print("\n--- Detailed Detection Information ---")
    for detection in detection_info['detections']:
        print(f"Plate {detection['id']}:")
        print(f"  - Confidence: {detection['confidence']:.3f}")
        print(f"  - Location: {detection['bbox']}")
        print(f"  - Area: {detection['area']} pixels")

    print("\n--- Saving Results ---")
    viz_path = detector.save_detection_visualization(test_image, results)
    plates = detector.extract_plate_regions(test_image, results)

    for i, plate in enumerate(plates):
        plate_filename = f"extracted_plate_{i+1}.jpg"
        plate_path = os.path.join(EXTRACTED_DIR, plate_filename)
        cv2.imwrite(plate_path, plate['image'])
        print(f"✓ Extracted plate saved: {plate_path}")

        # --- Otsu preprocess and save to processed dir ---
        processed_img = otsu_preprocess(plate['image'])
        processed_path = os.path.join(processed_dir, plate_filename)
        cv2.imwrite(processed_path, processed_img)
        print(f"✓ Otsu-preprocessed plate saved: {processed_path}")

    print(f"\n✅ Detection and preprocessing completed successfully!")
    print(f"📁 Visualizations: data/outputs/detected/")
    print(f"📁 Extracted plates: data/outputs/extracted/")
    print(f"📁 Otsu-preprocessed: data/outputs/processed/")

    return True

if __name__ == "__main__":
    success = test_license_plate_detection()
    if not success:
        print("\n🔧 Troubleshooting Tips:")
        print("1. Verify your model file and path.")
        print("2. Check if it's trained for license plate detection.")
        print("3. Try different confidence thresholds in the detector.")
        print("4. Ensure your test image contains visible license plates.")
