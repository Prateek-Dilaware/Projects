import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# --- PARAMETERS ---
plate_path = r"D:\code\Projects_ML_DL\NumberPlateDetector\data\outputs\extracted\WhatsApp Image 2025-07-26 at 07.39.43_e51a67a7_plate_1.jpg" #Image path 
min_height = 40
min_width = 200

# --- LOAD IMAGE ---
image = cv2.imread(plate_path)
assert image is not None, f"Could not load image: {plate_path}"

# --- GRAYSCALE ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- RESIZING ---
h, w = gray.shape
scale_factor = max(min_width / w, min_height / h) if (w < min_width or h < min_height) else 1.0
if scale_factor > 1.0:
    gray_resized = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
else:
    gray_resized = gray.copy()

# --- CLAHE (Local Contrast) ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray_resized)

# --- GAUSSIAN BLUR ---
blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

# --- OTSU THRESHOLDING ---
_, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- ALTERNATIVE: Adaptive Thresholding ---
binary_adaptive = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
)

# --- OCR WITH EASYOCR ON EACH STEP ---
reader = easyocr.Reader(['en'], gpu=False)
processing_stages = [
    ("Original", cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
    ("Gray", gray),
    ("Gray Resized", gray_resized),
    ("CLAHE Enhanced", enhanced),
    ("Gaussian Blurred", blurred),
    ("Otsu Binary", binary_otsu),
    ("Adaptive Binary", binary_adaptive),
]
ocr_results = []
for title, img in processing_stages:
    if len(img.shape) == 2:  # Grayscale/Binary: convert for EasyOCR
        ocr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        ocr_img = img
    result = reader.readtext(ocr_img)
    numbers = " ".join([res[1] for res in result]) if result else ""
    ocr_results.append((title, numbers))

# --- VISUALIZE ALL STAGES + OCR RESULTS ---
plt.figure(figsize=(15, 8))
for i, (title, img) in enumerate(processing_stages):
    plt.subplot(2, 4, i + 1)
    cmap = 'gray' if len(img.shape) == 2 else None
    plt.imshow(img, cmap=cmap)
    plt.title(f"{title}\nOCR: {ocr_results[i][1]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
