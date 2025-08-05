import cv2
import matplotlib.pyplot as plt

def preprocess_for_ocr_with_visuals(image):
    steps = {}

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    steps['Grayscale'] = gray.copy()

    # 2. Resize to ensure minimum character size
    h, w = gray.shape
    min_height, min_width = 40, 200
    scale_factor = max(min_width / w, min_height / h) if (w < min_width or h < min_height) else 1.0
    if scale_factor > 1.0:
        gray = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    steps['Resized'] = gray.copy()

    # 3. CLAHE (Local Contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    steps['CLAHE Enhanced'] = enhanced.copy()

    # 4. Gaussian Blur
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    steps['Blurred'] = blurred.copy()

    # 5. Otsu Thresholding
    otsu_thresh, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps['Otsu Thresholded'] = binary_otsu.copy()

    # 6. Adjusted Otsu (reduce threshold slightly to allow more ink-like density)
    adjusted_thresh = max(0, otsu_thresh - 15)
    _, binary_adjusted = cv2.threshold(blurred, adjusted_thresh, 255, cv2.THRESH_BINARY)
    steps['Adjusted Otsu (-15)'] = binary_adjusted.copy()

    return steps

def visualize_preprocessing(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at: {image_path}")
        return

    steps = preprocess_for_ocr_with_visuals(image)

    # Plotting
    plt.figure(figsize=(18, 10))
    for idx, (title, img) in enumerate(steps.items()):
        plt.subplot(2, 4, idx + 1)
        cmap = 'gray' if len(img.shape) == 2 else None
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = r"D:\code\Projects_ML_DL\NumberPlateDetector\data\outputs\extracted\WhatsApp Image 2025-07-26 at 07.39.43_e51a67a7_plate_1.jpg"  #  image path
    visualize_preprocessing(image_path)
