# License Plate Recognition System 🚘

This project captures real-time video from a webcam, detects license plates using Haar Cascades, and recognizes the text using EasyOCR.

## Features
- Real-time plate detection
- Text recognition with EasyOCR
- Image saving + CSV logging
- Dynamic detection area control

## Requirements
- Python 3.8+
- OpenCV
- EasyOCR
- NumPy

## How to Run
```bash
pip install -r requirements.txt
python main.py
```

## Keyboard Controls
- `s` — Save detected plate
- `+/-` — Adjust detection area
- `q` — Quit the application

## Sample Output
![Sample Plate](scanned_plates/sample.jpg)
