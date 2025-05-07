# Backend Context for ANPR System

This document summarizes the structure, dependencies, and functionality of the backend code for the Automatic Number Plate Recognition (ANPR) system.

## Overview

The backend is primarily a Python-based system designed to process video files to detect vehicles, track them, detect their license plates, and perform Optical Character Recognition (OCR) to read the plate numbers.

**Key Finding:** The current implementation (`src/main.py`) processes a hardcoded video file and outputs results to a CSV file. **It does not currently expose a web API endpoint** to handle image uploads from a frontend application.

## Core Components & Logic (`src/main.py`)

1.  **Model Loading**:
    *   Loads a pre-trained YOLOv8 model (`yolov8n.pt`) for general vehicle detection.
    *   Loads a custom-trained YOLOv8 model (`runs/detect/train2/weights/best.pt`) specifically for license plate detection.
2.  **Video Processing**:
    *   Opens and reads a specific video file (`../test_data/expressed_2103099-uhd_3840_2160_30fps.mp4`) frame by frame using OpenCV.
3.  **Vehicle Detection & Tracking**:
    *   Detects vehicles (classes `[2, 3, 5, 7]`) in each frame using the general YOLO model.
    *   Tracks detected vehicles across frames using the SORT algorithm (`sort.sort`).
4.  **License Plate Detection**:
    *   Detects license plates in each frame using the custom-trained YOLO model.
5.  **Association & Cropping**:
    *   Associates detected license plates with tracked vehicles based on spatial overlap (`util.get_car`).
    *   Crops the license plate region from the frame using OpenCV.
6.  **Image Preprocessing**:
    *   Converts the cropped plate to grayscale and applies binary thresholding using OpenCV (`cv2.cvtColor`, `cv2.threshold`).
7.  **OCR**:
    *   Uses EasyOCR (`easyocr.Reader`) to read the text from the preprocessed license plate crop (`util.read_license_plate`).
    *   Includes logic to format the recognized text and check against predefined formats (including a specific check for Indian license plates) (`util.format_license`, `util.license_complies_format`).
8.  **Output**:
    *   Stores results (frame number, car ID, bounding boxes, license text, scores) in a dictionary.
    *   Writes the collected results to a CSV file (`src/test3.csv`) using a custom function (`util.write_csv`).

## Key Dependencies

*   **`ultralytics`**: YOLOv8 object detection framework.
*   **`opencv-python`**: Computer vision library for image/video processing.
*   **`easyocr`**: Library for Optical Character Recognition.
*   **`numpy`**: Numerical computation.
*   **`sort-tracker`**: (Assumed, based on `from sort.sort import *`) Simple Online and Realtime Tracking algorithm implementation.
*   **`torch` / `torchvision`**: Deep learning framework (underlying dependency for `ultralytics` and potentially `easyocr`).
*   **(Other packages listed in `installed_packages_before.txt` support these core libraries or were potentially used during development/experimentation, e.g., `pandas`, `matplotlib`, `jupyter`*)**

## Potential Next Steps for Frontend Integration

To integrate with the React frontend, the backend needs significant modification:

1.  **API Framework**: Implement a web server using a framework like Flask or FastAPI.
2.  **API Endpoint**: Create an endpoint (e.g., `/detect`) that accepts POST requests with image data (file uploads).
3.  **Image Handling**: Modify the logic to process a single uploaded image instead of a video file. This involves adapting the detection and OCR steps for static images.
4.  **Response Format**: Return the detection results (number plate text, confidence) as a JSON response to the frontend.
5.  **Dependency Management**: Create a proper `requirements.txt` file listing only the necessary runtime dependencies. 