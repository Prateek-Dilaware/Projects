import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from modules.detector import LicensePlateDetector
from modules.easy_ocr_reader import EasyOCRReader
from main import otsu_preprocess
from config import INPUT_DIR, EXTRACTED_DIR, DETECTED_DIR, PROCESSED_DIR

st.set_page_config(page_title="License Plate Recognition", layout="wide")
st.title("🔎 License Plate Recognition (YOLO + Otsu + EasyOCR)")

# --- Upload image(s) and save to INPUT_DIR ---
uploaded_files = st.file_uploader(
    "Upload single/multiple images (JPEG, PNG, BMP)", 
    type=["jpg", "jpeg", "png", "bmp"], 
    accept_multiple_files=True
)

# Ensure input and output folders exist
for directory in [INPUT_DIR, EXTRACTED_DIR, PROCESSED_DIR, DETECTED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Clear INPUT_DIR before use (to avoid accidental re-processing)
for f in os.listdir(INPUT_DIR):
    try:
        os.remove(os.path.join(INPUT_DIR, f))
    except Exception:
        pass

input_image_paths = []
for file in uploaded_files:
    img_bytes = file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    save_path = os.path.join(INPUT_DIR, file.name)
    cv2.imwrite(save_path, img)
    input_image_paths.append(save_path)

# Instantiate detector and OCR once
detector = LicensePlateDetector()
ocr = EasyOCRReader(['en'])

results = []
if input_image_paths:
    for img_path in input_image_paths:
        img_name = os.path.basename(img_path)
        st.markdown(f"---\n### Input Image: `{img_name}`")
        img = cv2.imread(img_path)
        st.image(img, caption="Original", use_container_width=True)

        # --- Detection and Visualization ---
        results_detect = detector.detect(img_path)
        vis_path = detector.save_detection_visualization(img_path, results_detect)
        if vis_path and os.path.exists(vis_path):
            detected_vis_img = cv2.imread(vis_path)
            if detected_vis_img is not None:
                st.image(detected_vis_img, caption="Detection Visualization", use_container_width=True)
            else:
                st.warning("Could not load detection visualization image.")
        else:
            st.warning("No detection visualization generated.")

        # --- Cropping, Otsu, OCR ---
        plates = detector.extract_plate_regions(img_path, results_detect)
        if not plates:
            st.warning("No plates detected in this image.")
        for i, plate_data in enumerate(plates):
            crop_fname = f"{os.path.splitext(img_name)[0]}_plate_{i+1}.jpg"
            crop_path = os.path.join(EXTRACTED_DIR, crop_fname)
            cv2.imwrite(crop_path, plate_data["image"])
            st.image(plate_data["image"], caption=f"Cropped Plate {i+1}", channels="BGR", width=220)

            # Otsu processing (binarized)
            otsu_img = otsu_preprocess(plate_data["image"])
            otsu_path = os.path.join(PROCESSED_DIR, crop_fname)
            cv2.imwrite(otsu_path, otsu_img)
            st.image(otsu_img, caption=f"Otsu-Processed Plate {i+1}", width=220)

            # OCR
            plate_number = ocr.extract_text(otsu_img)
            st.success(f"Number Detected (Plate {i+1}): `{plate_number}`")

            results.append({
                "Input Image": img_name,
                "Plate Index": i + 1,
                "Cropped Plate": crop_fname,
                "Detected Number": plate_number
            })

    if results:
        st.markdown("### Download All Results as CSV")
        df = pd.DataFrame(results)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="plate_recognition_results.csv",
            mime="text/csv"
        )
else:
    st.info("Upload one or more images to begin.")
