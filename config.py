import os

# Model Configuration
MODEL_PATH = "models/license_plate_detector.pt"  # Path to detection model weights
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Paths
INPUT_DIR = "data/input/"
OUTPUT_DIR = "data/outputs/"
DETECTED_DIR = "data/outputs/detected/"
EXTRACTED_DIR = "data/outputs/extracted/"   # Raw cropped plate regions
PROCESSED_DIR = "data/outputs/processed/"   # Otsu-processed plates (for OCR input)

# Processing Settings
INPUT_SIZE = 640
SAVE_RESULTS = True
SHOW_RESULTS = True

# Create output directories if missing
os.makedirs(DETECTED_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
