import cv2
import csv
import os
import math
import time
from collections import defaultdict, deque
from ultralytics import YOLO


# ============================================================
# PERSON 2 - FINAL TRACKING + MOTION ANALYSIS
# ============================================================

MODEL_FILE = "yolo11n.pt"

OUTPUT_DIR = "output"

CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "final_tracking_data.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_movement_report.txt"
)

VIDEO_FILE = os.path.join(
    OUTPUT_DIR,
    "final_tracking.mp4"
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.4

CAMERA_ID = 0

HISTORY_LENGTH = 40

# Approximate calibration.
# Change this later after proper camera calibration.
PIXELS_PER_METER = 80

# Movement threshold
MOVEMENT_THRESHOLD = 5

# Speed considered unusually high
ABNORMAL_SPEED = 2.0

# Direction change detection
DIRECTION_CHANGE_THRESHOLD = 45


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ============================================================
# CAMERA SETTINGS
# ============================================================

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 30


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

video_writer = cv2.VideoWriter(
    VIDEO_FILE,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# CSV
# ============================================================

csv_file = open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow([
    "Timestamp",
    "Frame",
    "Person_ID",
    "X",
    "Y",
    "Direction",
    "Speed_mps",
    "Movement_Status",
    "Alert"
])


# ============================================================
# TRACKING STORAGE
# ============================================================

history = defaultdict(
    lambda: deque(
        maxlen=HISTORY_LENGTH
    )
)

previous_position = {}

previous_direction = {}

previous_time = {}

person_statistics = defaultdict(
    lambda: {
        "frames": 0,
        "moving": 0,
        "stationary": 0,
        "abnormal": 0,
        "total_distance": 0,
        "max_speed": 0,
        "directions": defaultdict(int)
    }
)


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_distance(p1, p2):

    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


def calculate_direction(previous, current):

    if previous is None:

        return "Stationary"

    dx = current[0] - previous[0]

    dy = current[1] - previous[1]

    if (
        abs(dx) < MOVEMENT_THRESHOLD
        and
        abs(dy) < MOVEMENT_THRESHOLD
    ):

        return "Stationary"

    if abs(dx) > abs(dy):

        if dx > 0:

            return "Right"

        return "Left"

    else:

        if dy > 0:

            return "Down"

        return "Up"


def direction_changed(old, new):

    if old is None:

        return False

    if old == "Stationary":

        return False

    if new == "Stationary":

        return False

    return old != new


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print("     PERSON 2 - FINAL TRACKING MODULE")
print("==============================================")
print()
print("Features enabled:")
print("  [✓] Person Detection")
print("  [✓] Person ID")
print("  [✓] Movement History")
print("  [✓] Direction")
print("  [✓] Speed")
print("  [✓] Stationary Detection")
print("  [✓] Movement Detection")
print("  [✓] Abnormal Speed Detection")
print("  [✓] Direction Change Detection")
print("  [✓] CSV Logging")
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

        print("Camera frame could not be read.")

        break


    frame_number += 1

    current_time = time.time()


    # --------------------------------------------------------
    # YOLO TRACK
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE,
        tracker="bytetrack.yaml",
        verbose=False
    )


    # --------------------------------------------------------
    # DETECTIONS
    # --------------------------------------------------------

    if (
        results
        and
        results[0].boxes is not None
        and
        results[0].boxes.id is not None
    ):

        boxes = results[0].boxes

        ids = boxes.id.int().cpu().tolist()

        coordinates = (
            boxes.xyxy
            .cpu()
            .tolist()
        )


        for box, person_id in zip(
            coordinates,
            ids
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )


            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

            history[person_id].append(
                center
            )


            old_position = (
                previous_position.get(
                    person_id
                )
            )


            # ------------------------------------------------
            # DIRECTION
            # ------------------------------------------------

            direction = calculate_direction(
                old_position,
                center
            )


            old_direction = (
                previous_direction.get(
                    person_id
                )
            )


            # ------------------------------------------------
            # SPEED
            # ------------------------------------------------

            speed_mps = 0

            if old_position is not None:

                distance_pixels = calculate_distance(
                    old_position,
                    center
                )

                old_time = (
                    previous_time.get(
                        person_id
                    )
                )

                if old_time is not None:

                    dt = (
                        current_time -
                        old_time
                    )

                    if dt > 0:

                        speed_pixels = (
                            distance_pixels /
                            dt
                        )

                        speed_mps = (
                            speed_pixels /
                            PIXELS_PER_METER
                        )


            # ------------------------------------------------
            # MOVEMENT STATUS
            # ------------------------------------------------

            if direction == "Stationary":

                status = "Stationary"

            else:

                status = "Moving"


            # ------------------------------------------------
            # ABNORMAL MOVEMENT
            # ------------------------------------------------

            alerts = []


            if speed_mps > ABNORMAL_SPEED:

                alerts.append(
                    "HIGH SPEED"
                )


            if direction_changed(
                old_direction,
                direction
            ):

                alerts.append(
                    "DIRECTION CHANGE"
                )


            if alerts:

                alert_text = ", ".join(
                    alerts
                )

            else:

                alert_text = "Normal"


            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            stats = person_statistics[
                person_id
            ]

            stats["frames"] += 1

            if status == "Moving":

                stats["moving"] += 1

            else:

                stats["stationary"] += 1


            if alert_text != "Normal":

                stats["abnormal"] += 1


            if old_position is not None:

                stats["total_distance"] += (
                    calculate_distance(
                        old_position,
                        center
                    )
                )


            if speed_mps > stats["max_speed"]:

                stats["max_speed"] = speed_mps


            stats["directions"][
                direction
            ] += 1


            # ------------------------------------------------
            # UPDATE PREVIOUS DATA
            # ------------------------------------------------

            previous_position[
                person_id
            ] = center

            previous_direction[
                person_id
            ] = direction

            previous_time[
                person_id
            ] = current_time


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # PERSON ID
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Person {person_id}",
                (x1, y1 - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DIRECTION
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Direction: {direction}",
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )


            # ------------------------------------------------
            # SPEED
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Speed: {speed_mps:.2f} m/s",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Status: {status}",
                (x1, y2 + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # ALERT
            # ------------------------------------------------

            if alert_text != "Normal":

                cv2.putText(
                    frame,
                    f"ALERT: {alert_text}",
                    (x1, y2 + 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            cv2.circle(
                frame,
                center,
                5,
                (0, 0, 255),
                -1
            )


            # ------------------------------------------------
            # MOVEMENT TRAIL
            # ------------------------------------------------

            points = history[
                person_id
            ]

            for i in range(
                1,
                len(points)
            ):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (255, 0, 0),
                    2
                )


            # ------------------------------------------------
            # SAVE CSV
            # ------------------------------------------------

            csv_writer.writerow([
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                frame_number,
                person_id,
                center[0],
                center[1],
                direction,
                round(
                    speed_mps,
                    2
                ),
                status,
                alert_text
            ])


    # ========================================================
    # GLOBAL DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        "PERSON 2 - TRACKING + MOTION",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Press Q to quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE VIDEO
    # ========================================================

    video_writer.write(frame)


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Person 2 - Final Tracking",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

video_writer.release()

csv_file.close()

cv2.destroyAllWindows()


# ============================================================
# GENERATE FINAL REPORT
# ============================================================

report = []

report.append(
    "=============================================="
)

report.append(
    "       PERSON 2 - FINAL MOVEMENT REPORT"
)

report.append(
    "=============================================="
)

report.append("")


report.append(
    f"Total unique tracked IDs: "
    f"{len(person_statistics)}"
)

report.append("")


for person_id, stats in person_statistics.items():

    report.append(
        "----------------------------------------------"
    )

    report.append(
        f"PERSON {person_id}"
    )

    report.append(
        "----------------------------------------------"
    )

    report.append(
        f"Total observations: "
        f"{stats['frames']}"
    )

    report.append(
        f"Moving observations: "
        f"{stats['moving']}"
    )

    report.append(
        f"Stationary observations: "
        f"{stats['stationary']}"
    )

    report.append(
        f"Abnormal observations: "
        f"{stats['abnormal']}"
    )

    report.append(
        f"Total movement: "
        f"{stats['total_distance']:.2f} pixels"
    )

    report.append(
        f"Maximum speed: "
        f"{stats['max_speed']:.2f} m/s"
    )


    if stats["directions"]:

        dominant_direction = max(
            stats["directions"],
            key=stats["directions"].get
        )

    else:

        dominant_direction = "Unknown"


    report.append(
        f"Dominant direction: "
        f"{dominant_direction}"
    )

    report.append("")


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print(
    "\n".join(report)
)

print()

print("==============================================")
print("       PERSON 2 MODULE COMPLETED")
print("==============================================")

print()

print(
    f"CSV: {CSV_FILE}"
)

print(
    f"Video: {VIDEO_FILE}"
)

print(
    f"Report: {REPORT_FILE}"
)

print()