import cv2
import os
from datetime import datetime
import easyocr

# Paths
MODEL_PATH = "haarcascade_russian_plate_number.xml"
SAVE_DIR = "scanned_plates"
LOG_FILE = os.path.join(SAVE_DIR, "plates_log.csv")

# Create necessary directories
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Load Haar Cascade
plate_cascade = cv2.CascadeClassifier(MODEL_PATH)
if plate_cascade.empty():
    print("Error loading Haar Cascade model.")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Minimum area threshold
min_area = 500
count = 0
last_recognized_number = ""

# Initialize log file if needed
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as file:
        file.write("Timestamp,File Path,Recognized Text\n")

# Open log file for appending
log_file_handle = open(LOG_FILE, "a")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    cv2.putText(img, f"Plates Detected: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    recognized_text = ""

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            img_roi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", img_roi)

            result = reader.readtext(img_roi, detail=0)
            if result:
                recognized_text = " ".join(result)
                last_recognized_number = recognized_text
                cv2.putText(img, f"Detected: {recognized_text}", (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Result", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and 'img_roi' in locals():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SAVE_DIR, f"plate_{timestamp}.jpg")
        cv2.imwrite(save_path, img_roi)
        print(f"Plate saved to {save_path}")
        log_file_handle.write(f"{timestamp},{save_path},{recognized_text}\n")
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1

    elif key == ord('+'):
        min_area += 100
        print(f"Min area increased to {min_area}")

    elif key == ord('-'):
        min_area = max(100, min_area - 100)
        print(f"Min area decreased to {min_area}")

    elif key == ord('q'):
        print("Exiting...")
        break

if last_recognized_number:
    print(f"Last recognized plate: {last_recognized_number}")
else:
    print("No plate detected.")

cap.release()
cv2.destroyAllWindows()
log_file_handle.close()
