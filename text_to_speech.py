import cv2
import pytesseract
from pytesseract import Output
import pyttsx3
import numpy as np

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 'c' to capture image and speak text, 'q' to quit.")

def preprocess_for_ocr(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Denoising
    gray = cv2.fastNlMeansDenoising(gray, h=30)
    
    # Adaptive thresholding (corrected)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    
    # Morphology to enhance text
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed

def ocr_text(image):
    # Tesseract config for sparse or multiple text lines
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text.strip()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        processed = preprocess_for_ocr(frame)
        
        # OCR on original + inverted image for best results
        text1 = ocr_text(processed)
        text2 = ocr_text(cv2.bitwise_not(processed))
        
        final_text = text1 + "\n" + text2
        final_text = "\n".join([line for line in final_text.split("\n") if line.strip()])
        
        print("Detected Text:\n", final_text)
        if final_text:
            engine.say(final_text)
            engine.runAndWait()
        else:
            print("No text detected!")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
