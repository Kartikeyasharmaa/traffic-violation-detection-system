# Traffic Violation Detection System

A modular Python project that detects:

- bike riders without helmets
- vehicles crossing a stop line during a simulated red light
- vehicles moving in the wrong direction

The system uses OpenCV and Ultralytics YOLO for video analysis, OCR for number-plate extraction, SQLite for persistence, FastAPI for APIs, and a static HTML/CSS/JS dashboard for administration.

## Project Structure

```text
traffic-violation-system/
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── models.py
│   ├── routes/
│   │   └── violations.py
│   └── static/images/
├── detection/
│   ├── helmet_detection.py
│   ├── ocr.py
│   ├── red_light_detection.py
│   ├── utils.py
│   └── wrong_side_detection.py
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── logs/
├── models/
├── config.py
└── requirements.txt
```

## Features

- Stores all violations in SQLite with image path, number plate, type, and timestamp.
- Saves evidence frames with unique filenames.
- Allows each detector to run independently from the command line.
- Provides a FastAPI backend with filtering, sorting, stats, and delete APIs.
- Serves a responsive admin dashboard with statistics cards and evidence thumbnails.
- Uses EasyOCR by default with pytesseract fallback.
- Writes runtime logs into the `logs/` folder.

## Requirements

- Python 3.10+
- A video file for each module
- Optional custom helmet model at `models/helmet.pt`
- Internet access the first time YOLO downloads `yolov8n.pt`, unless you place the model manually in `models/`
- If you use pytesseract fallback, install the Tesseract OCR binary on your system
- EasyOCR is optional. The project can run with `pytesseract` only, which avoids OpenCV GUI conflicts on some Windows setups.

## Installation

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional: place YOLO weights in the `models/` folder.

   - `models/yolov8n.pt` for general object detection
   - `models/helmet.pt` for best helmet-detection results

## Run the Backend and Dashboard

1. Start the API server:

   ```bash
   uvicorn backend.app:app --reload
   ```

   Important:

   - Run this command from the project root folder.
   - If you are already inside the `backend` folder, use:

   ```bash
   uvicorn app:app --reload
   ```

2. Open the dashboard:

   - http://127.0.0.1:8000/

3. API docs:

   - http://127.0.0.1:8000/docs

## Run Detection Modules

### 1. Helmet Detection

```bash
python -m detection.helmet_detection --video path\to\helmet_video.mp4
```

Optional arguments:

- `--vehicle-model models\yolov8n.pt`
- `--helmet-model models\helmet.pt`
- `--confidence 0.35`
- `--show` to display a live OpenCV window if your system supports it
- `--output outputs\helmet\result.mp4` to choose the annotated output file

Note: if `models/helmet.pt` is not present, the script falls back to a simple head-shape heuristic. A custom helmet model will give much better accuracy.

### 2. Red Light Jumping Detection

```bash
python -m detection.red_light_detection --video path\to\red_light_video.mp4 --line-y 0.65 --red-duration 8 --green-duration 10
```

Optional arguments:

- `--approach-direction top_to_bottom`
- `--model models\yolov8n.pt`
- `--confidence 0.35`
- `--show` to display a live OpenCV window if your system supports it
- `--output outputs\red_light\result.mp4` to choose the annotated output file

### 3. Wrong Side Driving Detection

```bash
python -m detection.wrong_side_detection --video path\to\wrong_side_video.mp4 --allowed-direction ltr
```

Optional arguments:

- `--allowed-direction ltr|rtl|ttb|btt`
- `--min-displacement 70`
- `--model models\yolov8n.pt`
- `--confidence 0.35`
- `--show` to display a live OpenCV window if your system supports it
- `--output outputs\wrong_side\result.mp4` to choose the annotated output file

## Camera Demo Files

If you need a live webcam demo, use these modules:

```bash
python -m detection.helmet_camera --camera 0 --show
python -m detection.red_light_camera --camera 0 --show
python -m detection.wrong_side_camera --camera 0 --show
```

Notes:

- Camera runs use the same detection pipeline and save annotated evidence the same way.
- For camera use, red-light line position and wrong-side direction may need small adjustment depending on camera placement.

## API Usage

### Get all violations

```bash
curl http://127.0.0.1:8000/violations
```

### Filter by type

```bash
curl "http://127.0.0.1:8000/violations?type=helmet"
```

### Delete a record

```bash
curl -X DELETE http://127.0.0.1:8000/violations/1
```

### Get statistics

```bash
curl http://127.0.0.1:8000/stats
```

## Database Schema

Table: `violations`

- `id` - primary key
- `violation_type` - `helmet`, `red_light`, or `wrong_side`
- `number_plate` - extracted OCR text
- `image_path` - relative path under `backend/static/images`
- `timestamp` - UTC timestamp of record creation

## How Data Flows

1. A detection script reads a video frame.
2. YOLO detects vehicles or riders.
3. Tracking logic decides whether the motion or rider behavior is a violation.
4. OCR attempts to read the number plate from the vehicle crop.
5. The system saves the evidence image in `backend/static/images/<violation_type>/`.
6. The annotated processed video is saved in `outputs/<violation_type>/`.
7. The record is inserted into SQLite.
8. The dashboard fetches data from the FastAPI backend.

## Notes

- Helmet detection works best with a dedicated custom helmet model.
- OCR accuracy depends heavily on video quality, camera angle, and plate visibility.
- The bundled centroid tracker keeps the project lightweight and avoids extra tracker dependencies.
