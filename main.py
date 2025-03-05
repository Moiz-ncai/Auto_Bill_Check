import os
import cv2
import re
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load YOLOv11 model
model = YOLO("weights/best.pt")  # Replace with your custom weights if needed

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Define class names
class_names = ["bill", "dates", "ref_num"]

# Define class-specific colors
class_colors = {
    0: (0, 255, 0),  # Green for 'bill'
    1: (255, 0, 0),  # Blue for 'dates'
    2: (0, 0, 255),  # Red for 'ref_num'
}


# Function to validate reference number format
def validate_ref_num(text):
    # Remove any special characters and spaces
    cleaned_text = re.sub(r'[^0-9]', '', text)

    # Check if the format is 14 digits followed by 'U'
    if re.match(r'^\d{14}$', cleaned_text):
        return cleaned_text
    return "N/A"


# Function to draw bounding boxes with OCR text
def draw_bboxes_with_ocr(image, boxes):
    bill_text, ref_num_text = "N/A", "N/A"

    # Group boxes by class and select the highest confidence detection for each class
    class_detections = {}
    for box in boxes:
        class_id = int(box.cls.item())
        conf = box.conf.item()

        # Only process 'bill' (0) and 'ref_num' (2) classes
        if class_id not in [0, 2]:
            continue

        # Keep the detection with the highest confidence for each class
        if class_id not in class_detections or conf > class_detections[class_id]['conf']:
            class_detections[class_id] = {
                'box': box,
                'conf': conf
            }

    # Process the selected detections
    for class_id, detection in class_detections.items():
        box = detection['box']

        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to integers

        # Crop the detected region for OCR
        cropped_obj = image[y1:y2, x1:x2]

        # Run OCR on the cropped image
        result = ocr.ocr(cropped_obj, cls=True)

        # Ensure OCR result is valid
        detected_texts = []
        if result and isinstance(result, list):  # Check if result is not None and is a list
            detected_texts = [line[1][0] for res in result if res for line in res]

        # Process the detected text based on class
        if class_id == 2 and detected_texts:
            # Validate reference number format
            detected_text = detected_texts[0]  # Keep only the first detected line for 'ref_num'
            ref_num_text = validate_ref_num(detected_text)
        elif class_id == 0:
            detected_text = " ".join(detected_texts).strip() if detected_texts else "N/A"
            bill_text = detected_text
        else:
            detected_text = "N/A"

        # Get class color
        color = class_colors.get(class_id, (255, 255, 255))  # Default to white

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw black rectangle behind text for better visibility
        text_size = cv2.getTextSize(detected_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x, text_y = x1, y1 - 10
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0),
                      -1)

        # Put OCR text on the image in white
        cv2.putText(image, detected_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw summary text in the top-left corner
    summary_text = f"Reference Number: {ref_num_text}\nTotal Bill: {bill_text} PKR"
    cv2.rectangle(image, (10, 10), (500, 70), (0, 0, 0), -1)  # Black background
    y_offset = 30
    for line in summary_text.split("\n"):
        cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    return image


# Process PDF file
def process_pdf(pdf_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Process each page
    for page_num in range(len(pdf_document)):
        # Render page to an image
        page = pdf_document[page_num]

        # Use scaling method you provided
        zoom_x = 2.0  # Scale factor for x-axis
        zoom_y = 2.0  # Scale factor for y-axis
        mat = fitz.Matrix(zoom_x, zoom_y)  # Create transformation matrix
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to an image using Pillow
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert Pillow image to OpenCV format (BGR)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Run inference
        results = model(img_cv, conf=0.2)  # Returns a list of Results objects

        # Process results
        for result in results:
            # Get bounding boxes
            boxes = result.boxes

            # Draw bounding boxes with OCR text and summary
            output_image = draw_bboxes_with_ocr(img_cv, boxes)

        # Save the output image
        output_path = os.path.join(output_folder, f"page_{page_num + 1}.jpg")
        cv2.imwrite(output_path, output_image)
        print(f"Processed and saved: {output_path}")

    # Close the PDF
    pdf_document.close()


# Main function
if __name__ == "__main__":
    pdf_path = "input.pdf"  # Replace with your PDF file path
    output_folder = "Output_Images_OCR"  # Replace with your output folder path

    process_pdf(pdf_path, output_folder)