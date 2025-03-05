import os
import sys
import cv2
import re
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QWidget, QScrollArea, QProgressBar,
    QSplitter, QFrame, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import (
    QPixmap, QImage, QIcon, QPalette, QColor,
    QLinearGradient, QPainter, QFont
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize


class StyledButton(QPushButton):
    def __init__(self, text, icon=None):
        super().__init__(text)

        # Set a modern, clean font
        font = QFont("Segoe UI", 10)
        self.setFont(font)

        # Style the button
        self.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)

        # Set icon if provided
        if icon:
            self.setIcon(QIcon(icon))
            self.setIconSize(QSize(20, 20))


class SideBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Sidebar styling
        self.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                color: white;
                padding: 10px;
            }
        """)

        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # File Selection
        self.file_path_label = QLabel("No PDF selected")
        self.file_path_label.setStyleSheet("""
            color: #ecf0f1;
            font-size: 10px;
            margin-bottom: 10px;
        """)
        layout.addWidget(self.file_path_label)

        # Buttons
        self.select_pdf_btn = StyledButton("Select PDF", "icons/open.png")
        self.prev_btn = StyledButton("Previous Page", "icons/previous.png")
        self.next_btn = StyledButton("Next Page", "icons/next.png")
        self.process_current_btn = StyledButton("Process Page", "icons/process.png")
        self.process_all_btn = StyledButton("Process All Pages", "icons/process_all.png")

        # Add buttons to layout
        layout.addWidget(self.select_pdf_btn)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.process_current_btn)
        layout.addWidget(self.process_all_btn)

        # Page info
        self.page_label = QLabel("Page: 0/0")
        self.page_label.setStyleSheet("""
            color: #bdc3c7;
            font-size: 10px;
            margin-top: 10px;
        """)
        layout.addWidget(self.page_label)

        # Add stretch to push items to top
        layout.addStretch(1)


class ProcessPagesThread(QThread):
    update_progress = pyqtSignal(int)
    processing_complete = pyqtSignal()

    def __init__(self, pdf_document, model, ocr):
        super().__init__()
        self.pdf_document = pdf_document
        self.model = model
        self.ocr = ocr
        self.processed_pages = {}

    def run(self):
        total_pages = len(self.pdf_document)
        for page_num in range(total_pages):
            # Render page
            page = self.pdf_document[page_num]
            img_cv = self.render_page(page)

            # Run inference
            results = self.model(img_cv, conf=0.2)

            # Process results
            for result in results:
                # Get bounding boxes
                boxes = result.boxes

                # Draw bounding boxes with OCR text and summary
                output_image, bill_text, ref_num_text = self.draw_bboxes_with_ocr(img_cv, boxes)

            # Convert processed image back to QPixmap
            output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            h, w, ch = output_image_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(output_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # Store processed page
            self.processed_pages[page_num] = pixmap

            # Update progress
            progress = int((page_num + 1) / total_pages * 100)
            self.update_progress.emit(progress)

        # Signal processing is complete
        self.processing_complete.emit()

    def render_page(self, page):
        # Render page to an image with better quality
        render_width = 1000  # Adjust as needed
        render_height = int(page.rect.height * render_width / page.rect.width)

        # Create transformation matrix to scale the page
        zoom = render_width / page.rect.width
        mat = fitz.Matrix(zoom, zoom)

        # Render pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to NumPy array
        img_cv = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3),
                              cv2.COLOR_RGB2BGR)

        return img_cv

    def draw_bboxes_with_ocr(self, image, boxes):
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
            result = self.ocr.ocr(cropped_obj, cls=True)

            # Ensure OCR result is valid
            detected_texts = []
            if result and isinstance(result, list):  # Check if result is not None and is a list
                detected_texts = [line[1][0] for res in result if res for line in res]

            # Process the detected text based on class
            if class_id == 2 and detected_texts:
                # Validate reference number format
                detected_text = detected_texts[0]  # Keep only the first detected line for 'ref_num'
                ref_num_text = self.validate_ref_num(detected_text)
            elif class_id == 0:
                detected_text = " ".join(detected_texts).strip() if detected_texts else "N/A"
                bill_text = detected_text
            else:
                detected_text = "N/A"

            # Get class color
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw black rectangle behind text for better visibility
            text_size = cv2.getTextSize(detected_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x, text_y = x1, y1 - 10
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
                          (0, 0, 0), -1)

            # Put OCR text on the image in white
            cv2.putText(image, detected_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw summary text in the top-left corner
        summary_text = f"Reference Number: {ref_num_text}\nTotal Bill: {bill_text} PKR"
        cv2.rectangle(image, (10, 10), (500, 70), (0, 0, 0), -1)  # Black background
        y_offset = 30
        for line in summary_text.split("\n"):
            cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        return image, bill_text, ref_num_text

    def validate_ref_num(self, text):
        # Remove any special characters and spaces
        cleaned_text = re.sub(r'[^0-9]', '', text)

        # Check if the format is 14 digits followed by 'U'
        if re.match(r'^\d{14}$', cleaned_text):
            return cleaned_text
        return "N/A"


class PDFOCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional PDF OCR Detection")
        self.setGeometry(100, 100, 1200, 800)

        # Main widget
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create sidebar
        self.sidebar = SideBar()

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 10px;
                margin: 0.5px;
            }
        """)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()

        # Image Display Area
        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        # Main content layout (scroll area + progress bar)
        content_layout = QVBoxLayout()
        content_layout.addWidget(self.scroll_area)
        content_layout.addWidget(self.progress_bar)

        content_widget = QWidget()
        content_widget.setLayout(content_layout)

        # Add sidebar and content to main layout
        main_layout.addWidget(self.sidebar, 1)
        main_layout.addWidget(content_widget, 4)

        # Connect sidebar buttons
        self.sidebar.select_pdf_btn.clicked.connect(self.select_pdf)
        self.sidebar.prev_btn.clicked.connect(self.previous_page)
        self.sidebar.next_btn.clicked.connect(self.next_page)
        self.sidebar.process_current_btn.clicked.connect(self.process_current_page)
        self.sidebar.process_all_btn.clicked.connect(self.process_all_pages)

        # Initialize rest of the application as before
        self.pdf_document = None
        self.current_page_num = 0
        self.total_pages = 0
        self.processed_pages = {}
        self.original_pages = {}

        # Load models
        self.load_models()

        # Update initial button states
        self.update_navigation_buttons()

        # Apply overall window styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
        """)

    def load_models(self):
        try:
            self.model = YOLO("weights/best.pt")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        except Exception as e:
            print(f"Error loading models: {e}")
            self.show_error_message("Could not load detection/OCR models")

    def process_all_pages(self):
        if not self.pdf_document:
            return

        # Disable buttons during processing
        self.disable_buttons_during_processing()

        # Create processing thread
        self.processing_thread = ProcessPagesThread(self.pdf_document, self.model, self.ocr)
        self.processing_thread.update_progress.connect(self.update_progress_bar)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)

        # Show progress bar
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        # Start processing
        self.processing_thread.start()

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def on_processing_complete(self):
        # Update processed pages
        self.processed_pages = self.processing_thread.processed_pages

        # Hide progress bar
        self.progress_bar.hide()

        # Re-enable buttons
        self.enable_buttons_after_processing()

        # Redisplay current page
        self.display_page()

    def disable_buttons_during_processing(self):
        self.sidebar.prev_btn.setEnabled(False)
        self.sidebar.next_btn.setEnabled(False)
        self.sidebar.process_current_btn.setEnabled(False)
        self.sidebar.process_all_btn.setEnabled(False)
        self.sidebar.select_pdf_btn.setEnabled(False)

    def enable_buttons_after_processing(self):
        self.update_navigation_buttons()
        self.sidebar.select_pdf_btn.setEnabled(True)

    def select_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf)")
        if file_path:
            # Use the sidebar's file path label
            self.sidebar.file_path_label.setText(file_path)
            self.open_pdf(file_path)

    def open_pdf(self, pdf_path):
        try:
            # Reset stored pages when a new PDF is opened
            self.processed_pages.clear()
            self.original_pages.clear()

            self.pdf_document = fitz.open(pdf_path)
            self.total_pages = len(self.pdf_document)
            self.current_page_num = 0
            self.update_navigation_buttons()
            self.display_page()
        except Exception as e:
            print(f"Error opening PDF: {e}")
            self.show_error_message("Could not open PDF file")

    def render_page(self, page):
        # Render page to an image with better quality
        # Use a fixed size rendering to avoid distortion
        render_width = 1000  # Adjust as needed
        render_height = int(page.rect.height * render_width / page.rect.width)

        # Create transformation matrix to scale the page
        zoom = render_width / page.rect.width
        mat = fitz.Matrix(zoom, zoom)

        # Render pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to NumPy array
        img_cv = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3),
                              cv2.COLOR_RGB2BGR)

        return img_cv

    def display_page(self):
        if not self.pdf_document:
            return

        # Check if page is already processed
        if self.current_page_num in self.processed_pages:
            # Display processed page
            pixmap = self.processed_pages[self.current_page_num]
        else:
            # Render original page if not processed
            page = self.pdf_document[self.current_page_num]

            # Store original page if not already stored
            if self.current_page_num not in self.original_pages:
                self.original_pages[self.current_page_num] = self.render_page(page)

            # Convert to QImage
            original_img = self.original_pages[self.current_page_num]
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h, w, ch = original_img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(original_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

        # Update page label using sidebar's page label
        self.sidebar.page_label.setText(f"Page: {self.current_page_num + 1}/{self.total_pages}")

        # Display image
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def process_current_page(self):
        if not self.pdf_document:
            return

        page = self.pdf_document[self.current_page_num]

        # Get the original page image
        if self.current_page_num not in self.original_pages:
            img_cv = self.render_page(page)
        else:
            img_cv = self.original_pages[self.current_page_num]

        # Run inference
        results = self.model(img_cv, conf=0.2)
        # Process results
        for result in results:
            # Get bounding boxes
            boxes = result.boxes

            # Draw bounding boxes with OCR text and summary
            output_image, bill_text, ref_num_text = self.draw_bboxes_with_ocr(img_cv, boxes)

        # Convert processed image back to QImage for display
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        h, w, ch = output_image_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(output_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Store the processed page
        self.processed_pages[self.current_page_num] = pixmap

        # Display processed image
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def next_page(self):
        if self.current_page_num < self.total_pages - 1:
            self.current_page_num += 1
            self.display_page()
            self.update_navigation_buttons()

    def previous_page(self):
        if self.current_page_num > 0:
            self.current_page_num -= 1
            self.display_page()
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        # Disable previous button on first page
        self.sidebar.prev_btn.setEnabled(self.current_page_num > 0)

        # Disable next button on last page
        self.sidebar.next_btn.setEnabled(
            self.current_page_num < self.total_pages - 1 if self.pdf_document else False
        )

        # Disable process buttons if no PDF is loaded
        self.sidebar.process_current_btn.setEnabled(self.pdf_document is not None)
        self.sidebar.process_all_btn.setEnabled(self.pdf_document is not None)

        # Update page label
        self.sidebar.page_label.setText(f"Page: {self.current_page_num + 1}/{self.total_pages}")

    def validate_ref_num(self, text):
        # Remove any special characters and spaces
        cleaned_text = re.sub(r'[^0-9]', '', text)

        # Check if the format is 14 digits followed by 'U'
        if re.match(r'^\d{14}$', cleaned_text):
            return cleaned_text
        return "N/A"

    def draw_bboxes_with_ocr(self, image, boxes):
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
            result = self.ocr.ocr(cropped_obj, cls=True)

            # Ensure OCR result is valid
            detected_texts = []
            if result and isinstance(result, list):  # Check if result is not None and is a list
                detected_texts = [line[1][0] for res in result if res for line in res]

            # Process the detected text based on class
            if class_id == 2 and detected_texts:
                # Validate reference number format
                detected_text = detected_texts[0]  # Keep only the first detected line for 'ref_num'
                ref_num_text = self.validate_ref_num(detected_text)
            elif class_id == 0:
                detected_text = " ".join(detected_texts).strip() if detected_texts else "N/A"
                bill_text = detected_text
            else:
                detected_text = "N/A"

            # Get class color
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw black rectangle behind text for better visibility
            text_size = cv2.getTextSize(detected_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x, text_y = x1, y1 - 10
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
                          (0, 0, 0), -1)

            # Put OCR text on the image in white
            cv2.putText(image, detected_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw summary text in the top-left corner
        summary_text = f"Reference Number: {ref_num_text}\nTotal Bill: {bill_text} PKR"
        cv2.rectangle(image, (10, 10), (500, 70), (0, 0, 0), -1)  # Black background
        y_offset = 30
        for line in summary_text.split("\n"):
            cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        return image, bill_text, ref_num_text

    def show_error_message(self, message):
        # Simple error handling (you could use QMessageBox for a more robust solution)
        print(f"Error: {message}")


def main():
    app = QApplication(sys.argv)

    app.setFont(QFont("Segoe UI", 10))

    main_window = PDFOCRApp()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
