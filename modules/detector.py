from ultralytics import YOLO
import cv2
import os
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, DETECTED_DIR

class LicensePlateDetector:
    def __init__(self, model_path=None):
        """Initialize the license plate detector."""
        self.model_path = model_path or MODEL_PATH

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        try:
            self.model = YOLO(self.model_path)
            print(f"✓ Model loaded successfully from: {self.model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

    def detect(self, image, conf_threshold=None):
        """
        Detect license plates in an image.
        Accepts: file path (str) or numpy array (BGR image).
        """
        conf = conf_threshold if conf_threshold is not None else CONFIDENCE_THRESHOLD

        # YOLOv8 ultralytics: can accept filepath or ndarray, pass-through
        try:
            results = self.model(image, conf=conf)
            return results
        except Exception as e:
            print(f"Detection failed: {e}")
            return None

    def get_detection_info(self, results):
        """Extract detailed detection information."""
        if not results:
            return {"total_plates": 0, "detections": []}

        detection_info = {"total_plates": 0, "detections": []}
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                detection_info["total_plates"] += len(boxes)
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    detection_info["detections"].append({
                        "id": i + 1,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence,
                        "area": (x2 - x1) * (y2 - y1)
                    })
        return detection_info

    def save_detection_visualization(self, image, results, output_name=None):
        """
        Save image with detection bounding boxes.
        image: file path (str) or numpy array (BGR image)
        """
        if not results or not getattr(results[0], "boxes", None):
            print("No detections to visualize")
            return None

        # Create visualization with YOLO's .plot(), which returns an annotated image (numpy array)
        annotated_image = results[0].plot()
        # Determine output file name from input
        if output_name is None:
            if isinstance(image, str):
                base_name = os.path.splitext(os.path.basename(image))[0]
                output_name = f"{base_name}_detected.jpg"
            else:
                output_name = "detected_yolo.jpg"  # fallback for arrays
        output_path = os.path.join(DETECTED_DIR, output_name)
        os.makedirs(DETECTED_DIR, exist_ok=True)
        cv2.imwrite(output_path, annotated_image)
        print(f"✓ Detection visualization saved: {output_path}")
        return output_path

    def extract_plate_regions(self, image, results):
        """
        Extract cropped license plate regions as images.
        image: file path (str) or numpy array (BGR image).
        """
        if not results:
            return []

        # Load or assign image as appropriate
        if isinstance(image, str):
            original_image = cv2.imread(image)
        else:
            original_image = image.copy()
        if original_image is None:
            print(f"Failed to load/process image: {image}")
            return []

        plates = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_plate = original_image[y1:y2, x1:x2]
                    plates.append({
                        "image": cropped_plate,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(box.conf[0])
                    })
        return plates
