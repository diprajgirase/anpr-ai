import os
import sys
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to find utils
# Adjust the path based on where api.py is located relative to util.py
# Assuming api.py is in backend/src and util.py is also in backend/src
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir) # Add src directory
sys.path.append(os.path.dirname(current_dir)) # Add backend directory if needed for models/sort

# Import utility functions AFTER adjusting sys.path
# Use a try-except block for robustness
try:
    from util import read_license_plate, format_license, license_complies_format
except ImportError as e:
    logger.error(f"Error importing from util: {e}")
    # Define dummy functions if import fails, to allow Flask app to start
    def read_license_plate(crop):
        return None, None
    def license_complies_format(text):
        return False
    def format_license(text):
        return text

app = Flask(__name__)
# Explicitly allow requests from the frontend origin
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# --- Configuration ---
# Adjust model paths as needed, relative to this script's location or use absolute paths
MODEL_PATH = os.path.join(os.path.dirname(current_dir), 'runs', 'detect', 'train2', 'weights', 'best.pt')
EASYOCR_LANGUAGES = ['en']
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence for YOLO detection
OCR_THRESHOLD = 0.1 # Minimum confidence for EasyOCR result

# --- Test Route --- 
@app.route('/', methods=['GET'])
def health_check():
    logger.info("GET / received (health check)")
    return jsonify({"status": "Backend is running"}), 200

# --- Model Loading ---
def load_models():
    """Load models on startup."""
    plate_detector = None
    ocr_reader = None
    try:
        if os.path.exists(MODEL_PATH):
            plate_detector = YOLO(MODEL_PATH)
            logger.info(f"YOLO license plate detector loaded successfully from {MODEL_PATH}")
        else:
            logger.error(f"YOLO model file not found at: {MODEL_PATH}")
        
        ocr_reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=False) # Set gpu=True if CUDA is available and configured
        logger.info(f"EasyOCR reader loaded successfully for languages: {EASYOCR_LANGUAGES}")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
    return plate_detector, ocr_reader

license_plate_detector, reader = load_models()

# --- API Endpoint ---
@app.route('/detect', methods=['POST'])
def detect_plate():
    if license_plate_detector is None or reader is None:
        logger.error("Models not loaded. Cannot process request.")
        return jsonify({"error": "Models not loaded on server"}), 500

    if 'image' not in request.files:
        logger.warning("No image file found in request")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read image file into memory
        in_memory_file = BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
        
        # Decode image using OpenCV
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            return jsonify({"error": "Invalid image format or corrupted file"}), 400
            
        logger.info("Image decoded successfully.")

        # Detect license plates
        results = license_plate_detector(img, conf=CONFIDENCE_THRESHOLD)[0] # Apply confidence threshold here
        logger.info(f"YOLO detection completed. Found {len(results.boxes)} potential boxes.")

        best_result = {"numberPlate": None, "confidence": 0.0}

        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            logger.info(f"Detected Box: [({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f})] Score: {score:.4f}")

            # Crop license plate
            license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
            
            if license_plate_crop.size == 0:
                logger.warning(f"Skipping empty crop for box: [({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f})]")
                continue

            # Preprocess license plate (optional, EasyOCR might handle it)
            # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            
            # Read license plate number using EasyOCR
            # Pass the color crop directly, EasyOCR handles preprocessing
            ocr_results = reader.readtext(license_plate_crop)
            logger.info(f"OCR attempted on crop. Found {len(ocr_results)} text segments.")

            for ocr_det in ocr_results:
                bbox, text, ocr_score = ocr_det
                text = text.upper().replace(' ', '')
                logger.info(f"  OCR Result: Text='{text}', Score={ocr_score:.4f}")

                # ---> ADDED LOGGING HERE <---
                logger.info(f"    Checking format for cleaned text: '{text}'") 
                is_valid_format = license_complies_format(text)
                logger.info(f"    Result from license_complies_format: {is_valid_format}")
                # --------------------------

                # Check if format complies and score is above threshold
                # if ocr_score > OCR_THRESHOLD and license_complies_format(text):
                if ocr_score > OCR_THRESHOLD and is_valid_format: # Use the stored result
                    formatted_text = format_license(text)
                    logger.info(f"    Valid Plate Found: '{formatted_text}' (Formatted), Score: {ocr_score:.4f}")
                    # Use OCR score as confidence for the final result
                    if ocr_score > best_result["confidence"]:
                        best_result["numberPlate"] = formatted_text
                        best_result["confidence"] = ocr_score
                else:
                    # logger.info(f"    Plate '{text}' rejected (Score: {ocr_score:.4f}, Format Valid: {license_complies_format(text)})")
                    # Use the stored result in the log message for clarity
                    logger.info(f"    Plate '{text}' rejected (Score: {ocr_score:.4f}, Format Valid: {is_valid_format})")

        if best_result["numberPlate"]:
             logger.info(f"Final Best Result: {best_result}")
             return jsonify(best_result)
        else:
             logger.info("No valid license plate found after OCR and formatting.")
             return jsonify({"error": "No valid license plate detected"}), 404

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return jsonify({"error": "Internal server error processing image"}), 500

if __name__ == '__main__':
    # Run the Flask app
    # Ensure models are loaded before starting
    if license_plate_detector is None or reader is None:
        logger.critical("CRITICAL: Models failed to load. Flask app cannot start properly.")
        sys.exit(1) # Exit if models aren't loaded
        
    logger.info("Starting Flask server...")
    # Use host='0.0.0.0' to make it accessible on your network
    # Changed port from 5000 to 5001
    app.run(host='0.0.0.0', port=5001, debug=False) # TRYING PORT 5001 