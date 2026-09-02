import cv2
import csv
import os
import math
import time
from collections import defaultdict, deque
from ultralytics import YOLO


# ============================================================
# PERSON 2 - COMPLETE TRACKING MODULE
# ============================================================

# Load YOLO model
model = YOLO("yolo11n.pt")


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CONFIDENCE = 0.4

# Approximate pixels-per-meter.
# This is an estimation because a normal camera does not
# provide real-world distance automatically.
PIXELS_PER_METER = 80

# Number of previous positions stored for each person
HISTORY_LENGTH = 30

# Camera
CAMERA_ID = 0


# ------------------------------------------------------------
# CREATE OUTPUT FOLDER
# ------------------------------------------------------------

os.makedirs("output", exist_ok=True)


# ------------------------------------------------------------
# CSV FILE FOR TRACKING DATA
# ------------------------------------------------------------

csv_file = "output/tracking_data.csv"

csv_exists = os.path.exists(csv_file)

csv_output = open(
    csv_file,
    "a",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_output)

if not csv_exists:
    csv_writer.writerow([
        "Timestamp",
        "Frame",
        "Person_ID",
        "X",
        "Y",
        "Direction",
        "Speed_pixels_per_second",
        "Speed_meters_per_second"
    ])


# ------------------------------------------------------------
# OPEN CAMERA
# ------------------------------------------------------------

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    csv_output.close()
    exit()


# ------------------------------------------------------------
# CAMERA INFORMATION
# ------------------------------------------------------------

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


# ------------------------------------------------------------
# OUTPUT VIDEO
# ------------------------------------------------------------

output_video = "output/person_tracking.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (frame_width, frame_height)
)


# ------------------------------------------------------------
# MOVEMENT HISTORY
# ------------------------------------------------------------

movement_history = defaultdict(
    lambda: deque(maxlen=HISTORY_LENGTH)
)


# ------------------------------------------------------------
# PREVIOUS TIME FOR SPEED CALCULATION
# ------------------------------------------------------------

previous_time = {}


# ------------------------------------------------------------
# PREVIOUS POSITION
# ------------------------------------------------------------

previous_position = {}


# ------------------------------------------------------------
# FUNCTION: CALCULATE DIRECTION
# ------------------------------------------------------------

def calculate_direction(previous, current):

    if previous is None:
        return "Stationary"

    dx = current[0] - previous[0]
    dy = current[1] - previous[1]

    # Ignore tiny movements caused by detection noise
    threshold = 5

    if abs(dx) < threshold and abs(dy) < threshold:
        return "Stationary"

    if abs(dx) > abs(dy):

        if dx > 0:
            return "Right"

        else:
            return "Left"

    else:

        if dy > 0:
            return "Down"

        else:
            return "Up"


# ------------------------------------------------------------
# FUNCTION: CALCULATE DISTANCE
# ------------------------------------------------------------

def calculate_distance(point1, point2):

    return math.sqrt(
        (point2[0] - point1[0]) ** 2 +
        (point2[1] - point1[1]) ** 2
    )


# ------------------------------------------------------------
# START TRACKING
# ------------------------------------------------------------

print()
print("==============================================")
print("      PERSON 2 - TRACKING MODULE")
print("==============================================")
print()
print("Camera started.")
print("Person IDs will appear automatically.")
print()
print("Features:")
print("  [✓] Person Detection")
print("  [✓] Person Identification")
print("  [✓] Movement History")
print("  [✓] Movement Direction")
print("  [✓] Speed Estimation")
print("  [✓] CSV Tracking Data")
print("  [✓] Output Video")
print()
print("Press Q to stop.")
print()


frame_number = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    frame_number += 1

    current_time = time.time()


    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[0],          # Class 0 = person
        conf=CONFIDENCE,
        tracker="bytetrack.yaml",
        verbose=False
    )


    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if results and results[0].boxes is not None:

        boxes = results[0].boxes

        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()

            coordinates = boxes.xyxy.cpu().tolist()

            confidences = boxes.conf.cpu().tolist()


            for box, person_id, confidence in zip(
                coordinates,
                track_ids,
                confidences
            ):

                x1, y1, x2, y2 = map(int, box)


                # ------------------------------------------------
                # CENTER POINT
                # ------------------------------------------------

                center_x = int((x1 + x2) / 2)

                center_y = int((y1 + y2) / 2)

                current_position = (
                    center_x,
                    center_y
                )


                # ------------------------------------------------
                # MOVEMENT HISTORY
                # ------------------------------------------------

                movement_history[person_id].append(
                    current_position
                )


                # ------------------------------------------------
                # PREVIOUS POSITION
                # ------------------------------------------------

                old_position = previous_position.get(
                    person_id
                )


                # ------------------------------------------------
                # MOVEMENT DIRECTION
                # ------------------------------------------------

                direction = calculate_direction(
                    old_position,
                    current_position
                )


                # ------------------------------------------------
                # SPEED CALCULATION
                # ------------------------------------------------

                speed_pixels = 0

                speed_meters = 0


                if old_position is not None:

                    distance_pixels = calculate_distance(
                        old_position,
                        current_position
                    )


                    old_time = previous_time.get(
                        person_id
                    )


                    if old_time is not None:

                        time_difference = (
                            current_time - old_time
                        )


                        if time_difference > 0:

                            speed_pixels = (
                                distance_pixels /
                                time_difference
                            )


                            speed_meters = (
                                speed_pixels /
                                PIXELS_PER_METER
                            )


                # Save current values
                previous_position[person_id] = (
                    current_position
                )

                previous_time[person_id] = (
                    current_time
                )


                # ------------------------------------------------
                # DRAW BOUNDING BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # ------------------------------------------------
                # PERSON LABEL
                # ------------------------------------------------

                label = (
                    f"Person {person_id}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                # ------------------------------------------------
                # DIRECTION LABEL
                # ------------------------------------------------

                direction_text = (
                    f"Direction: {direction}"
                )

                cv2.putText(
                    frame,
                    direction_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )


                # ------------------------------------------------
                # SPEED LABEL
                # ------------------------------------------------

                speed_text = (
                    f"Speed: {speed_meters:.2f} m/s"
                )

                cv2.putText(
                    frame,
                    speed_text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )


                # ------------------------------------------------
                # CENTER POINT
                # ------------------------------------------------

                cv2.circle(
                    frame,
                    current_position,
                    5,
                    (0, 0, 255),
                    -1
                )


                # ------------------------------------------------
                # DRAW MOVEMENT TRAIL
                # ------------------------------------------------

                history = movement_history[
                    person_id
                ]


                for i in range(
                    1,
                    len(history)
                ):

                    cv2.line(
                        frame,
                        history[i - 1],
                        history[i],
                        (255, 0, 0),
                        2
                    )


                # ------------------------------------------------
                # SAVE TRACKING DATA
                # ------------------------------------------------

                csv_writer.writerow([
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    frame_number,
                    person_id,
                    center_x,
                    center_y,
                    direction,
                    round(speed_pixels, 2),
                    round(speed_meters, 2)
                ])


    # --------------------------------------------------------
    # DISPLAY SYSTEM INFORMATION
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "PERSON TRACKING",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # SAVE VIDEO
    # --------------------------------------------------------

    video_writer.write(frame)


    # --------------------------------------------------------
    # SHOW CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "Person 2 - Tracking",
        frame
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print()
        print("Stopping tracking...")

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

video_writer.release()

csv_output.close()

cv2.destroyAllWindows()


print()
print("==============================================")
print("        TRACKING COMPLETED")
print("==============================================")
print()
print("Tracking data:")
print("  output/tracking_data.csv")
print()
print("Tracking video:")
print("  output/person_tracking.mp4")
print()
print("==============================================")