import easyocr
import re

class EasyOCRReader:
    """
    A class for OCR using EasyOCR. You can use this class in both scripts and the Streamlit UI.
    """
    def __init__(self, lang_list=['en'], gpu=False):
        self.reader = easyocr.Reader(lang_list, gpu=gpu)
        print(f"✅ EasyOCR initialized for languages: {lang_list} (GPU: {gpu})")

    def clean_text(self, text):
        # Uppercase and remove all non-alphanumeric characters
        out = re.sub(r'[^A-Z0-9]', '', text.upper())
        # Custom substitutions to fix common OCR confusions
        substitutions = {'O': '0', 'I': '1', 'S': '5', 'Q': '0'}
        # Apply substitutions after first 2 characters (which are usually state code like "MP", "DL", etc.)
        out = out[:2] + ''.join(substitutions.get(c, c) for c in out[2:])
        return out

    def extract_text(self, image):
        """
        Pass a grayscale or color plate image (NumPy array).
        Returns the cleaned license plate string (may be empty if detection failed).
        """
        result = self.reader.readtext(image)
        if result:
            extracted = []
            for _, text, _ in result:
                filtered = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(filtered) >= 3:  # Only keep meaningful segments
                    extracted.append(filtered)
            fulltext = ''.join(extracted)
            cleaned = self.clean_text(fulltext)
            print(f"✅ EasyOCR extracted segments: {extracted} → Combined: '{fulltext}' → Cleaned: '{cleaned}'")
            return cleaned
        else:
            print("❌ No text detected")
            return ""

# Optional standalone testing
if __name__ == "__main__":
    import cv2
    import os
    processed_dir = "data/outputs/processed/"
    reader = EasyOCRReader(['en'], gpu=False)  # class usage
    for plate_file in os.listdir(processed_dir):
        if plate_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
            img = cv2.imread(os.path.join(processed_dir, plate_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print(f"\nProcessing: {plate_file}")
                text = reader.extract_text(img)
                print(f"  Result: {text}")
