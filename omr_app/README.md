# Automated MCQ Grading - OMR

This project is an automated MCQ grading system using Optical Mark Recognition (OMR) and a Streamlit web interface. It allows you to upload scanned answer sheets, templates, and answer keys, and automatically grades student responses.

## Features

- Upload template, key, and student answer sheets
- Automatic detection of answer bubbles using Hough Circle Transform
- Robust row grouping with DBSCAN clustering
- Adaptive thresholding for marked/unmarked bubble detection
- Visual feedback: detected circles are shown on all sheets
- Per-question and total score reporting
- Easy-to-use Streamlit UI

## Project Structure

```
omr_app/
├── main.py            # Streamlit UI
├── omr_core.py        # OMR logic (preprocessing, detection, grouping, grading)
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/razilahamed/Automated-MCQ-Grading---OMR.git
   cd Automated-MCQ-Grading---OMR/omr_app
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run main.py
   ```
2. In the web UI:
   - Set the number of questions and options
   - Upload the template sheet (blank OMR), answer key sheet, and student sheet
   - Click 'Grade' to process and view results
   - See detected circles and grading feedback for each sheet

## How It Works

- **Template Calibration:** Detects bubbles in the template to calibrate circle size and spacing
- **Student/Key Detection:** Uses template parameters to robustly detect bubbles in student and key sheets
- **Row Grouping:** DBSCAN clusters bubbles into rows, tolerant to noise/skew
- **Mark Detection:** Adaptive thresholding based on template statistics
- **Scoring:** Compares student answers to key and reports results

## Troubleshooting

- If circles are not detected, check image quality and try adjusting the number of questions/options
- Make sure uploaded images are clear, high-contrast scans
- For best results, use the same template layout for all sheets
