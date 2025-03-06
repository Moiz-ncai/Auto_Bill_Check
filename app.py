import sys
import cv2
import re
import fitz  # PyMuPDF
import numpy as np
import csv
from ultralytics import YOLO
from paddleocr import PaddleOCR

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QWidget, QScrollArea, QProgressBar,
    QSplitter, QFrame, QGraphicsDropShadowEffect, QLineEdit, QMessageBox
)
from PyQt5.QtGui import (
    QPixmap, QImage, QIcon, QPalette, QColor,
    QLinearGradient, QPainter, QFont
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize


class StyledLineEdit(QLineEdit):
    def __init__(self, placeholder_text=""):
        super().__init__()

        # Set a modern, clean font
        font = QFont("Segoe UI", 10)
        self.setFont(font)

        # Style the line edit
        self.setStyleSheet("""
            QLineEdit {
                background-color: #34495e;
                color: white;
                border: 1px solid #2c3e50;
                padding: 5px;
                border-radius: 4px;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)

        # Set placeholder text
        self.setPlaceholderText(placeholder_text)


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
            QLabel {
                color: #ecf0f1;
                margin-bottom: 5px;
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

        # NEW: Add filter button for invalid reference numbers
        self.filter_invalid_btn = StyledButton("Show Invalid References", "icons/filter.png")
        self.filter_invalid_btn.setCheckable(True)  # Make it toggleable
        layout.addWidget(self.filter_invalid_btn)

        # Page info
        self.page_label = QLabel("Page: 0/0")
        self.page_label.setStyleSheet("""
            color: #bdc3c7;
            font-size: 10px;
            margin-top: 10px;
        """)
        layout.addWidget(self.page_label)

        # Results Section
        results_header = QLabel("Processing Results")
        results_header.setStyleSheet("""
            color: #ecf0f1;
            font-weight: bold;
            margin-top: 20px;
        """)
        layout.addWidget(results_header)

        # Reference Number Section
        ref_num_label = QLabel("Reference Number:")
        ref_num_label.setStyleSheet("""
            color: #ecf0f1;
            margin-bottom: 5px;
        """)
        layout.addWidget(ref_num_label)
        self.ref_num_edit = StyledLineEdit("Enter Reference Number")
        layout.addWidget(self.ref_num_edit)

        # Total Bill Section
        bill_label = QLabel("Total Bill (PKR):")
        bill_label.setStyleSheet("""
            color: #ecf0f1;
            margin-bottom: 5px;
        """)
        layout.addWidget(bill_label)
        self.bill_edit = StyledLineEdit("Enter Total Bill")
        layout.addWidget(self.bill_edit)
        self.save_csv_btn = StyledButton("Save to CSV", "icons/save.png")
        layout.addWidget(self.save_csv_btn)

        # Add stretch to push items to top
        layout.addStretch(1)


class ProcessPagesThread(QThread):
    update_progress = pyqtSignal(int)
    processing_complete = pyqtSignal()

    def __init__(self, pdf_document, model, ocr, result_signal):
        super().__init__()
        self.pdf_document = pdf_document
        self.model = model
        self.ocr = ocr
        self.result_signal = result_signal
        self.processed_pages = {}

    def run(self):
        total_pages = len(self.pdf_document)
        for page_num in range(total_pages):
            # Render page
            page = self.pdf_document[page_num]
            img_cv = self.render_page(page)

            # Default values in case no detection occurs
            bill_text = "N/A"
            ref_num_text = "N/A"
            ref_num_valid = True

            # Run inference
            results = self.model(img_cv, conf=0.2)

            # Process results
            for result in results:
                # Get bounding boxes
                boxes = result.boxes

                # Draw bounding boxes with OCR text and summary
                output_image, bill_text, ref_num_text, ref_num_valid = self.draw_bboxes_with_ocr(img_cv, boxes)

                # Emit results for this page using the passed signal
                self.result_signal.emit(page_num, ref_num_text, bill_text, ref_num_valid)

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
        ref_num_valid = True  # Default to valid

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
                ref_num_text, ref_num_valid = self.validate_ref_num(detected_text)
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

        return image, bill_text, ref_num_text, ref_num_valid
    def validate_ref_num(self, text):
        # Remove any special characters and spaces
        cleaned_text = re.sub(r'[^0-9]', '', text)

        # Check if the format is 14 digits
        is_valid = re.match(r'^\d{14}$', cleaned_text) is not None

        # Return both the original text and a validation flag
        return cleaned_text, is_valid


class PDFOCRApp(QMainWindow):
    page_result_signal = pyqtSignal(int, str, str, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional PDF OCR Detection")
        self.setGeometry(100, 100, 1400, 800)
        self.page_results = {}
        self.sidebar = SideBar()
        self.page_result_signal.connect(self.update_page_results)
        self.sidebar.ref_num_edit.textChanged.connect(self.validate_and_save_ref_num)
        self.sidebar.bill_edit.textChanged.connect(self.save_bill)
        self.sidebar.save_csv_btn.clicked.connect(self.save_results_to_csv)

        self.sidebar.filter_invalid_btn.clicked.connect(self.toggle_invalid_filter)

        self.show_only_invalid = False
        self.invalid_pages = []

        # Main widget
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

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

    def toggle_invalid_filter(self):
        """Toggle between showing all pages and showing only pages with invalid reference numbers"""
        self.show_only_invalid = self.sidebar.filter_invalid_btn.isChecked()

        if self.show_only_invalid:
            # Update the button text to indicate current state
            self.sidebar.filter_invalid_btn.setText("Show All Pages")

            # Find all pages with invalid reference numbers
            self.invalid_pages = [
                page_num for page_num, result in self.page_results.items()
                if not result.get('ref_num_valid', True)
            ]

            if not self.invalid_pages:
                # No invalid pages found
                QMessageBox.information(self, "No Invalid References",
                                        "No pages with invalid reference numbers were found.")
                self.sidebar.filter_invalid_btn.setChecked(False)
                self.show_only_invalid = False
                return

            # Navigate to the first invalid page
            if self.invalid_pages:
                self.current_page_num = self.invalid_pages[0]
                self.display_page()
                self.update_navigation_buttons()
        else:
            # Restore normal navigation
            self.sidebar.filter_invalid_btn.setText("Show Invalid References")
            self.invalid_pages = []
            self.update_navigation_buttons()

    def validate_and_save_ref_num(self, text):
        """Validate and save the reference number when it's edited"""
        if self.pdf_document:
            # Validate the reference number format
            cleaned_text, is_valid = self.validate_ref_num(text)

            # Get the previous validation state (if it exists)
            previous_state = self.page_results.get(self.current_page_num, {}).get('ref_num_valid', True)

            # Store the results
            self.page_results[self.current_page_num] = self.page_results.get(self.current_page_num, {})
            self.page_results[self.current_page_num]['ref_num'] = text
            self.page_results[self.current_page_num]['ref_num_valid'] = is_valid

            # If we're in filter mode and validation state changed, we might need to update the filter
            if self.show_only_invalid and previous_state != is_valid:
                if is_valid and self.current_page_num in self.invalid_pages:
                    # Page is now valid, remove from invalid list
                    self.invalid_pages.remove(self.current_page_num)
                    if not self.invalid_pages:
                        # No more invalid pages, turn off filter
                        QMessageBox.information(self, "No Invalid References",
                                                "All reference numbers are now valid!")
                        self.sidebar.filter_invalid_btn.setChecked(False)
                        self.show_only_invalid = False
                elif not is_valid and self.current_page_num not in self.invalid_pages:
                    # Page is now invalid, add to invalid list
                    self.invalid_pages.append(self.current_page_num)
                    self.invalid_pages.sort()  # Keep the list in order

            # Update the text box styling based on validation
            if is_valid:
                self.sidebar.ref_num_edit.setStyleSheet("""
                    QLineEdit {
                        background-color: #34495e;
                        color: white;
                        border: 1px solid #2c3e50;
                        padding: 5px;
                        border-radius: 4px;
                    }
                    QLineEdit:focus {
                        border: 1px solid #3498db;
                    }
                """)
            else:
                self.sidebar.ref_num_edit.setStyleSheet("""
                    QLineEdit {
                        background-color: #34495e;
                        color: white;
                        border: 1px solid #e74c3c;
                        padding: 5px;
                        border-radius: 4px;
                        background-color: rgba(231, 76, 60, 0.3);
                    }
                    QLineEdit:focus {
                        border: 1px solid #e74c3c;
                    }
                """)

            # Update navigation buttons in case filter state changed
            self.update_navigation_buttons()

    def save_bill(self, text):
        # Save the edited bill for the current page
        if self.pdf_document:
            self.page_results[self.current_page_num] = self.page_results.get(self.current_page_num, {})
            self.page_results[self.current_page_num]['bill'] = text

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
        self.processing_thread = ProcessPagesThread(
            self.pdf_document,
            self.model,
            self.ocr,
            self.page_result_signal  # Pass the signal to the thread
        )
        self.processing_thread.update_progress.connect(self.update_progress_bar)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)

        # Show progress bar
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        # Start processing
        self.processing_thread.start()

    def update_page_results(self, page_num, ref_num, bill, ref_num_valid=True):
        # Store results for specific page
        self.page_results[page_num] = {
            'ref_num': ref_num,
            'bill': bill,
            'ref_num_valid': ref_num_valid
        }

        # If this is the current page, update sidebar
        if page_num == self.current_page_num:
            self.sidebar.ref_num_edit.setText(ref_num)
            self.sidebar.bill_edit.setText(bill)

            # Update text box styling based on validation
            if ref_num_valid:
                self.sidebar.ref_num_edit.setStyleSheet("""
                    QLineEdit {
                        background-color: #34495e;
                        color: white;
                        border: 1px solid #2c3e50;
                        padding: 5px;
                        border-radius: 4px;
                    }
                    QLineEdit:focus {
                        border: 1px solid #3498db;
                    }
                """)
            else:
                self.sidebar.ref_num_edit.setStyleSheet("""
                    QLineEdit {
                        background-color: #34495e;
                        color: white;
                        border: 1px solid #e74c3c;
                        padding: 5px;
                        border-radius: 4px;
                        background-color: rgba(231, 76, 60, 0.3);
                    }
                    QLineEdit:focus {
                        border: 1px solid #e74c3c;
                    }
                """)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def update_sidebar_results(self, ref_num, bill):
        # Update sidebar labels from processing thread
        self.sidebar.ref_num_label.setText(f"Reference Number: {ref_num}")
        self.sidebar.bill_label.setText(f"Total Bill: {bill} PKR")

    def on_processing_complete(self):
        # Update processed pages
        self.processed_pages = self.processing_thread.processed_pages

        # If we're in filter mode, update the invalid pages list
        if self.show_only_invalid:
            self.invalid_pages = [
                page_num for page_num, result in self.page_results.items()
                if not result.get('ref_num_valid', True)
            ]

            if not self.invalid_pages:
                # No invalid pages found after processing
                QMessageBox.information(self, "No Invalid References",
                                        "All processed pages have valid reference numbers.")
                self.sidebar.filter_invalid_btn.setChecked(False)
                self.show_only_invalid = False

        # Hide progress bar
        self.progress_bar.hide()

        # Re-enable buttons
        self.enable_buttons_after_processing()

        # Redisplay current page
        self.display_page()

        # Show summary of invalid pages
        invalid_count = len([1 for result in self.page_results.values() if not result.get('ref_num_valid', True)])
        if invalid_count > 0:
            QMessageBox.information(self, "Processing Complete",
                                    f"Processing complete. Found {invalid_count} pages with invalid reference numbers.")

    def save_results_to_csv(self):
        # Check if we have any processed results
        if not self.page_results:
            QMessageBox.warning(self, "No Results", "No processed pages found. Please process PDF pages first.")
            return

        # Open file dialog to choose save location
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    # Create CSV writer
                    csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

                    # Write header
                    csv_writer.writerow(['Page', 'Reference Number', 'Total Bill (PKR)'])

                    # Sort results by page number
                    sorted_pages = sorted(self.page_results.items())

                    # Write each page's results
                    for page_num, result in sorted_pages:
                        # Ensure reference number is treated as text
                        ref_num = result.get('ref_num', 'N/A')
                        ref_num = f"=\"{ref_num}\"" if ref_num != 'N/A' else 'N/A'

                        csv_writer.writerow([
                            page_num + 1,  # Add 1 to make page numbers more user-friendly
                            ref_num,
                            result.get('bill', 'N/A')
                        ])

                # Show success message
                QMessageBox.information(self, "Success", f"Results saved to {file_path}")

            except Exception as e:
                # Show error if saving fails
                QMessageBox.critical(self, "Error", f"Could not save CSV: {str(e)}")

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

        # Update results for current page
        if self.current_page_num in self.page_results:
            result = self.page_results[self.current_page_num]
            self.sidebar.ref_num_edit.setText(result.get('ref_num', 'N/A'))
            self.sidebar.bill_edit.setText(result.get('bill', 'N/A'))

            # Update text box styling based on validation
            if result.get('ref_num_valid', True):
                self.sidebar.ref_num_edit.setStyleSheet("""
                    QLineEdit {
                        background-color: #34495e;
                        color: white;
                        border: 1px solid #2c3e50;
                        padding: 5px;
                        border-radius: 4px;
                    }
                    QLineEdit:focus {
                        border: 1px solid #3498db;
                    }
                """)
            else:
                self.sidebar.ref_num_edit.setStyleSheet("""
                    QLineEdit {
                        background-color: #34495e;
                        color: white;
                        border: 1px solid #e74c3c;
                        padding: 5px;
                        border-radius: 4px;
                        background-color: rgba(231, 76, 60, 0.3);
                    }
                    QLineEdit:focus {
                        border: 1px solid #e74c3c;
                    }
                """)
        else:
            # Reset if no results for this page
            self.sidebar.ref_num_edit.setText('N/A')
            self.sidebar.bill_edit.setText('N/A')
            # Reset to default styling
            self.sidebar.ref_num_edit.setStyleSheet("""
                QLineEdit {
                    background-color: #34495e;
                    color: white;
                    border: 1px solid #2c3e50;
                    padding: 5px;
                    border-radius: 4px;
                }
                QLineEdit:focus {
                    border: 1px solid #3498db;
                }
            """)

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

        # Default values
        bill_text = "N/A"
        ref_num_text = "N/A"
        ref_num_valid = True

        # Process results
        for result in results:
            # Get bounding boxes
            boxes = result.boxes

            # Draw bounding boxes with OCR text and summary
            output_image, bill_text, ref_num_text, ref_num_valid = self.draw_bboxes_with_ocr(img_cv, boxes)

        # Only update if not already manually edited
        if self.current_page_num not in self.page_results or \
                self.page_results[self.current_page_num].get('ref_num') == 'N/A':
            # Store results for this page
            self.page_results[self.current_page_num] = {
                'ref_num': ref_num_text,
                'bill': bill_text,
                'ref_num_valid': ref_num_valid
            }

            # Update sidebar labels
            self.sidebar.ref_num_edit.setText(ref_num_text)
            self.sidebar.bill_edit.setText(bill_text)

            # Update text box styling based on validation
            if ref_num_valid:
                self.sidebar.ref_num_edit.setStyleSheet("""
                    QLineEdit {
                        background-color: #34495e;
                        color: white;
                        border: 1px solid #2c3e50;
                        padding: 5px;
                        border-radius: 4px;
                    }
                    QLineEdit:focus {
                        border: 1px solid #3498db;
                    }
                """)
            else:
                self.sidebar.ref_num_edit.setStyleSheet("""
                    QLineEdit {
                        background-color: #34495e;
                        color: white;
                        border: 1px solid #e74c3c;
                        padding: 5px;
                        border-radius: 4px;
                        background-color: rgba(231, 76, 60, 0.3);
                    }
                    QLineEdit:focus {
                        border: 1px solid #e74c3c;
                    }
                """)

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
        """Navigate to the next page, respecting the filter if active"""
        if not self.pdf_document:
            return

        if self.show_only_invalid and self.invalid_pages:
            # Find the next invalid page
            current_index = self.invalid_pages.index(
                self.current_page_num) if self.current_page_num in self.invalid_pages else -1
            if current_index < len(self.invalid_pages) - 1:
                self.current_page_num = self.invalid_pages[current_index + 1]
                self.display_page()
                self.update_navigation_buttons()
        else:
            # Normal navigation
            if self.current_page_num < self.total_pages - 1:
                self.current_page_num += 1
                self.display_page()
                self.update_navigation_buttons()

    def previous_page(self):
        """Navigate to the previous page, respecting the filter if active"""
        if not self.pdf_document:
            return

        if self.show_only_invalid and self.invalid_pages:
            # Find the previous invalid page
            current_index = self.invalid_pages.index(
                self.current_page_num) if self.current_page_num in self.invalid_pages else len(self.invalid_pages)
            if current_index > 0:
                self.current_page_num = self.invalid_pages[current_index - 1]
                self.display_page()
                self.update_navigation_buttons()
        else:
            # Normal navigation
            if self.current_page_num > 0:
                self.current_page_num -= 1
                self.display_page()
                self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update navigation buttons based on current page and filter state"""
        if self.show_only_invalid and self.invalid_pages:
            # When filtering, base navigation on the filtered list
            current_index = self.invalid_pages.index(
                self.current_page_num) if self.current_page_num in self.invalid_pages else -1
            self.sidebar.prev_btn.setEnabled(current_index > 0)
            self.sidebar.next_btn.setEnabled(current_index < len(self.invalid_pages) - 1)

            # Update page label to show both absolute page number and position in filtered list
            if current_index >= 0:
                self.sidebar.page_label.setText(
                    f"Page: {self.current_page_num + 1}/{self.total_pages} (Invalid: {current_index + 1}/{len(self.invalid_pages)})")
            else:
                self.sidebar.page_label.setText(f"Page: {self.current_page_num + 1}/{self.total_pages}")
        else:
            # Normal navigation logic
            self.sidebar.prev_btn.setEnabled(self.current_page_num > 0)
            self.sidebar.next_btn.setEnabled(
                self.current_page_num < self.total_pages - 1 if self.pdf_document else False)
            self.sidebar.page_label.setText(f"Page: {self.current_page_num + 1}/{self.total_pages}")

        # Process buttons should always be enabled if a PDF is loaded
        self.sidebar.process_current_btn.setEnabled(self.pdf_document is not None)
        self.sidebar.process_all_btn.setEnabled(self.pdf_document is not None)

    def validate_ref_num(self, text):
        # Remove any special characters and spaces
        cleaned_text = re.sub(r'[^0-9]', '', text)

        # Check if the format is 14 digits
        is_valid = re.match(r'^\d{14}$', cleaned_text) is not None

        # Return both the original text and a validation flag
        return cleaned_text, is_valid

    def draw_bboxes_with_ocr(self, image, boxes):
        bill_text = "N/A"
        ref_num_text = "N/A"
        ref_num_valid = True  # Default to valid

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
                temp_ref_num, temp_valid = self.validate_ref_num(detected_text)
                ref_num_text = temp_ref_num
                ref_num_valid = temp_valid
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

        # Explicitly return all 4 values
        return image, bill_text, ref_num_text, ref_num_valid

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
