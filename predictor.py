import os
import glob
from main import NumberPlateRecognitionSystem  # Ensure this is now based on EasyOCR pipeline

class BatchPredictor:
    def __init__(self):
        self.processor = NumberPlateRecognitionSystem()

    def predict_all_images(self, input_folder="data/input/"):
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        results = {}
        for extension in image_extensions:
            pattern = os.path.join(input_folder, extension)
            for image_path in glob.glob(pattern):
                filename = os.path.basename(image_path)
                print(f"\n--- Processing {filename} ---")
                extracted_texts = self.processor.process_single_image(image_path)
                results[filename] = extracted_texts
        return results

    def save_results(self, results, output_file="results.txt"):
        with open(output_file, 'w') as f:
            for filename, texts in results.items():
                f.write(f"{filename}: {texts}\n")

if __name__ == "__main__":
    predictor = BatchPredictor()
    all_results = predictor.predict_all_images()
    predictor.save_results(all_results)
