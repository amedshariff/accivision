# PERSON 1 — VIDEO + YOLO

Build:
1. OpenCV video reader/preprocessor.
2. YOLO object detector.
3. Annotated video writer.
4. Detection CSV/JSON exporter.

INPUT:
- input/raw/mock_normal.mp4
- input/raw/mock_accident.mp4
- input/config.json

EXPECTED OUTPUT:
- processed video
- annotated video
- detection CSV/JSON

HANDOFF TO PERSON 2:
mock_yolo_detections.csv

DO NOT:
- create tracking IDs in the detector
- call one frame an accident
- treat confidence as accident probability

Detection schema:
frame,timestamp_sec,class,x1,y1,x2,y2,confidence
