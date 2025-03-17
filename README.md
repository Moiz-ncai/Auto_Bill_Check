# Intelligent Bill Reader

An advanced desktop application for automated bill processing and reference number extraction from PDF documents.

## Overview

The Intelligent Bill Reader is a PyQt5-based application that uses computer vision and OCR (Optical Character Recognition) to automatically extract critical information from bill documents. It's designed to identify and validate reference numbers and total bill amounts from scanned PDF bills, making data entry more efficient and reducing human error.

## Features

- **PDF Support**: Open and navigate through multi-page PDF documents
- **AI-Powered Detection**: Uses YOLO object detection to locate reference numbers and bill amounts
- **OCR Integration**: Employs PaddleOCR for accurate text extraction from detected regions
- **Reference Number Validation**: Automatically validates 14-digit reference numbers according to required format
- **Batch Processing**: Process individual pages or entire documents with progress tracking
- **Invalid Reference Filter**: Easily find and navigate to pages with invalid or missing reference numbers
- **Export Capability**: Save extracted data to CSV for further analysis or record-keeping
- **Modern UI**: Clean, intuitive interface with customizable styling

## Installation

### Prerequisites

- Python 3.10
- PyQt5
- OpenCV
- PyMuPDF (fitz)
- Ultralytics YOLO
- PaddleOCR
- NumPy

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/Moiz-ncai/Auto_Bill_Check
   cd Auto_Bill_Check
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download model weights:
   - Place the YOLO model weights in the `weights/` directory
   - Make sure the file is named `best.pt`

4. Run the application:
   ```
   python main.py
   ```

## Usage

1. **Open PDF**: Click "Select PDF" to open a bill document
2. **Navigate**: Use the previous/next buttons to move between pages
3. **Process**: Click "Process Page" to analyze the current page or "Process Entire Folder" for batch processing
4. **Review**: Check the extracted reference number and bill amount
5. **Filter**: Toggle "Show Invalid References" to focus on pages with problematic reference numbers
6. **Edit**: Manually edit extracted data if needed
7. **Export**: Click "Save to CSV" to export all processed data

## Reference Number Validation

The application validates reference numbers according to the following rules:
- Must be exactly 14 digits
- Cannot contain letters or special characters
- Missing or improperly formatted reference numbers are flagged as invalid

## UI Components

- **Left Panel**: Contains controls and extracted data fields
- **Main Area**: Displays the current PDF page with detection overlays
- **Progress Bar**: Shows processing status during batch operations

## Implementation Details

The application uses a multi-threaded approach to keep the UI responsive during processing:
- Main thread handles UI interactions
- Worker thread performs AI detection and OCR processing
- Results are communicated back to the main thread via signals

## Requirements

```
PyQt5>=5.15.0
opencv-python>=4.5.0
PyMuPDF>=1.18.0
numpy>=1.19.0
ultralytics>=8.0.0
paddleocr>=2.0.0
```

## Directory Structure

```
intelligent-bill-reader/
├── main.py           # Main application file
├── weights/          # Model weights directory
│   └── best.pt       # YOLO model weights
├── icons/            # UI icons
│   ├── open.png
│   ├── process.png
│   └── ...
└── requirements.txt  # Dependencies
```

