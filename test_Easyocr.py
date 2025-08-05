import cv2
import os
import easyocr

def test_easyocr_ocr():
    print("🧪 EasyOCR Test with Otsu Preprocessing")
    print("=" * 40)
    processed_dir = "data/outputs/processed/"
    if not os.path.exists(processed_dir):
        print(f"❌ Directory not found: {processed_dir}")
        return
    plate_files = [
        f for f in os.listdir(processed_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
    ]
    print(f"Detected plate_files: {plate_files}")
    if not plate_files:
        print(f"❌ No plate images found in {processed_dir}")
        return

    reader = easyocr.Reader(['en'], gpu=False)

    successful = 0
    for plate_file in plate_files:
        print(f"\n📸 Testing: {plate_file}")
        plate_path = os.path.join(processed_dir, plate_file)
        image = cv2.imread(plate_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            result = reader.readtext(image)
            if result:
                texts = [item[1] for item in result]
                fulltext = ' '.join(texts)
                successful += 1
                print(f"  ✅ EasyOCR Otsu Result: '{fulltext}'")
            else:
                print("  ❌ No text detected")
        else:
            print(f"  ❌ Failed to load image file: {plate_file}")

    print(f"\n📈 SUMMARY")
    print("=" * 24)
    print(f"Total plates tested: {len(plate_files)}")
    print(f"Successful extractions: {successful}")
    print(f"Success rate: {((successful / len(plate_files)) * 100) if plate_files else 0:.1f}%")

if __name__ == "__main__":
    test_easyocr_ocr()
