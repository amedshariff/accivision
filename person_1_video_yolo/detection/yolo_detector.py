import cv2
import csv
import json

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PERSON 1 - YOLO OBJECT DETECTION
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# 2. INPUT VIDEO
# ------------------------------------------------------------

INPUT_VIDEO = (
    PROJECT_ROOT
    / "person_1_video_yolo"
    / "output"
    / "processed_road_test.mp4"
)


# ------------------------------------------------------------
# 3. OUTPUT FOLDER
# ------------------------------------------------------------

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "person_1_video_yolo"
    / "output"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 4. OUTPUT FILES
# ------------------------------------------------------------

ANNOTATED_VIDEO = (
    OUTPUT_FOLDER
    / "annotated_road_test.mp4"
)

DETECTIONS_CSV = (
    OUTPUT_FOLDER
    / "yolo_detections.csv"
)

DETECTIONS_JSON = (
    OUTPUT_FOLDER
    / "yolo_detections.json"
)


# ------------------------------------------------------------
# 5. YOLO MODEL
# ------------------------------------------------------------

MODEL_NAME = "yolo26n.pt"


# ------------------------------------------------------------
# 6. CONFIDENCE THRESHOLD
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.50


# ------------------------------------------------------------
# 7. ROAD-SAFETY CLASSES
# ------------------------------------------------------------

ROAD_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
}


# ------------------------------------------------------------
# 8. CHECK INPUT VIDEO
# ------------------------------------------------------------

if not INPUT_VIDEO.exists():

    print()
    print("ERROR: Processed video not found.")
    print()
    print("Expected:")
    print(INPUT_VIDEO)
    print()
    print("Run video_processor.py first.")

    raise SystemExit(1)


# ------------------------------------------------------------
# 9. LOAD YOLO MODEL
# ------------------------------------------------------------

print()
print("=" * 65)
print("PERSON 1 - YOLO OBJECT DETECTION")
print("=" * 65)

print()
print("Loading YOLO model...")
print("Model:", MODEL_NAME)

model = YOLO(MODEL_NAME)

print("YOLO model loaded successfully!")


# ------------------------------------------------------------
# 10. OPEN VIDEO
# ------------------------------------------------------------

cap = cv2.VideoCapture(
    str(INPUT_VIDEO)
)


if not cap.isOpened():

    print()
    print("ERROR: Could not open processed video.")

    raise SystemExit(1)


# ------------------------------------------------------------
# 11. READ VIDEO INFORMATION
# ------------------------------------------------------------

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print()
print("Video Information")
print("-------------------------")
print("FPS:", fps)
print("Width:", width)
print("Height:", height)
print("Total frames:", total_frames)


# ------------------------------------------------------------
# 12. CREATE ANNOTATED VIDEO
# ------------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(ANNOTATED_VIDEO),
    fourcc,
    fps,
    (width, height)
)


if not writer.isOpened():

    print()
    print("ERROR: Could not create annotated video.")

    cap.release()

    raise SystemExit(1)


# ------------------------------------------------------------
# 13. CREATE CSV
# ------------------------------------------------------------

csv_file = open(
    DETECTIONS_CSV,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(
    csv_file
)


# ------------------------------------------------------------
# 14. CSV HEADER
# ------------------------------------------------------------
#
# IMPORTANT:
# frame_id is used instead of fname.
#

csv_writer.writerow([
    "frame_id",
    "timestamp_sec",
    "class",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence"
])


# ------------------------------------------------------------
# 15. JSON DETECTIONS
# ------------------------------------------------------------

json_detections = []


# ------------------------------------------------------------
# 16. PROCESS VIDEO
# ------------------------------------------------------------

frame_id = 0


try:

    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        success, frame = cap.read()


        if not success:

            break


        # ----------------------------------------------------
        # INCREMENT FRAME NUMBER
        # ----------------------------------------------------

        frame_id += 1


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        if fps > 0:

            timestamp_sec = (
                (frame_id - 1) / fps
            )

        else:

            timestamp_sec = 0.0


        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )


        result = results[0]


        # ----------------------------------------------------
        # PROCESS DETECTIONS
        # ----------------------------------------------------

        for box in result.boxes:

            # ------------------------------------------------
            # CLASS ID
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )


            # ------------------------------------------------
            # CLASS NAME
            # ------------------------------------------------

            class_name = model.names[
                class_id
            ]


            # ------------------------------------------------
            # KEEP ROAD-SAFETY CLASSES
            # ------------------------------------------------

            if class_name not in ROAD_CLASSES:

                continue


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # CREATE DETECTION
            # ------------------------------------------------

            detection = {

                "frame_id": frame_id,

                "timestamp_sec": round(
                    timestamp_sec,
                    3
                ),

                "class": class_name,

                "x1": x1,

                "y1": y1,

                "x2": x2,

                "y2": y2,

                "confidence": round(
                    confidence,
                    4
                )
            }


            # ------------------------------------------------
            # SAVE TO CSV
            # ------------------------------------------------

            csv_writer.writerow([
                detection["frame_id"],
                detection["timestamp_sec"],
                detection["class"],
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
                detection["confidence"]
            ])


            # ------------------------------------------------
            # SAVE TO JSON LIST
            # ------------------------------------------------

            json_detections.append(
                detection
            )


        # ----------------------------------------------------
        # DRAW DETECTIONS
        # ----------------------------------------------------

        annotated_frame = result.plot()


        # ----------------------------------------------------
        # WRITE ANNOTATED VIDEO
        # ----------------------------------------------------

        writer.write(
            annotated_frame
        )


        # ----------------------------------------------------
        # DISPLAY VIDEO
        # ----------------------------------------------------

        cv2.imshow(
            "Person 1 - YOLO Detection",
            annotated_frame
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if frame_id % 1 == 0:

            print(
                f"Processed {frame_id}/{total_frames} frames"
            )


        # ----------------------------------------------------
        # PRESS Q TO STOP
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print()
            print("Processing stopped by user.")

            break


finally:

    cap.release()

    writer.release()

    csv_file.close()

    cv2.destroyAllWindows()


# ------------------------------------------------------------
# 17. SAVE JSON
# ------------------------------------------------------------

with open(
    DETECTIONS_JSON,
    "w",
    encoding="utf-8"
) as json_file:

    json.dump(
        json_detections,
        json_file,
        indent=4
    )


# ------------------------------------------------------------
# 18. FINAL OUTPUT
# ------------------------------------------------------------

print()
print("=" * 65)
print("YOLO DETECTION COMPLETED")
print("=" * 65)

print()
print("Frames processed:")
print(frame_id)

print()
print("Total detections:")
print(len(json_detections))

print()
print("Annotated video:")
print(ANNOTATED_VIDEO)

print()
print("CSV:")
print(DETECTIONS_CSV)

print()
print("JSON:")
print(DETECTIONS_JSON)

print()
print(" YOLO stage completed successfully.")

#--------------------------------------------------------------------------------------
#for Output run this below command in terminal
#python person_1_video_yolo/detection/yolo_detector.py   
#--------------------------------------------------------------------------------------