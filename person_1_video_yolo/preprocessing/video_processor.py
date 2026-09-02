import cv2
from pathlib import Path


# ============================================================
# PERSON 1
# OPENCV VIDEO PREPROCESSING
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_VIDEO = (
    PROJECT_ROOT
    / "person_1_video_yolo"
    / "input"
    / "raw"
    / "real_road_test.mp4"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "person_1_video_yolo"
    / "output"
)

OUTPUT_VIDEO = (
    OUTPUT_FOLDER
    / "processed_road_test.mp4"
)


# ------------------------------------------------------------
# 2. CREATE OUTPUT FOLDER
# ------------------------------------------------------------

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 3. CHECK INPUT VIDEO
# ------------------------------------------------------------

if not INPUT_VIDEO.exists():

    print()
    print("ERROR: Input video not found.")
    print()
    print("Expected:")
    print(INPUT_VIDEO)
    print()

    raise SystemExit(1)


# ------------------------------------------------------------
# 4. OPEN INPUT VIDEO
# ------------------------------------------------------------

cap = cv2.VideoCapture(
    str(INPUT_VIDEO)
)


if not cap.isOpened():

    print()
    print("ERROR: Could not open input video.")
    print()

    raise SystemExit(1)


# ------------------------------------------------------------
# 5. READ VIDEO INFORMATION
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


# ------------------------------------------------------------
# 6. CALCULATE DURATION
# ------------------------------------------------------------

if fps > 0:

    duration = total_frames / fps

else:

    duration = 0


# ------------------------------------------------------------
# 7. DISPLAY VIDEO INFORMATION
# ------------------------------------------------------------

print()
print("=" * 65)
print("PERSON 1 - OPENCV VIDEO PREPROCESSING")
print("=" * 65)

print()
print("Input video:")
print(INPUT_VIDEO)

print()
print("Video Information")
print("-------------------------")
print("FPS:", fps)
print("Width:", width)
print("Height:", height)
print("Frame count:", total_frames)
print("Duration:", round(duration, 2), "seconds")


# ------------------------------------------------------------
# 8. VALIDATE VIDEO PROPERTIES
# ------------------------------------------------------------

if fps <= 0:

    print()
    print("ERROR: Invalid FPS.")

    cap.release()

    raise SystemExit(1)


if width <= 0 or height <= 0:

    print()
    print("ERROR: Invalid video resolution.")

    cap.release()

    raise SystemExit(1)


# ------------------------------------------------------------
# 9. CREATE VIDEO WRITER
# ------------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


if not writer.isOpened():

    print()
    print("ERROR: Could not create output video.")

    cap.release()

    raise SystemExit(1)


# ------------------------------------------------------------
# 10. PROCESS VIDEO FRAME-BY-FRAME
# ------------------------------------------------------------

frame_number = 0


while True:

    # Read one frame

    success, frame = cap.read()


    # End of video

    if not success:

        break


    frame_number += 1


    # --------------------------------------------------------
    # FRAME PREPROCESSING
    # --------------------------------------------------------
    #
    # For this baseline pipeline, we preserve the original
    # frame without changing its resolution.
    #
    # This keeps the pixel coordinates consistent with YOLO.
    #

    processed_frame = frame


    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    writer.write(
        processed_frame
    )


    # --------------------------------------------------------
    # DISPLAY FRAME
    # --------------------------------------------------------

    cv2.imshow(
        "Person 1 - OpenCV Preprocessing",
        processed_frame
    )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_number % 20 == 0:

        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


    # --------------------------------------------------------
    # PRESS Q TO STOP
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print()
        print("Processing stopped by user.")

        break


# ------------------------------------------------------------
# 11. RELEASE RESOURCES
# ------------------------------------------------------------

cap.release()

writer.release()

cv2.destroyAllWindows()


# ------------------------------------------------------------
# 12. FINAL RESULT
# ------------------------------------------------------------

print()
print("=" * 65)
print("OPENCV PROCESSING COMPLETED")
print("=" * 65)

print()
print("Frames processed:", frame_number)

print()
print("Processed video:")
print(OUTPUT_VIDEO)

print()
print("OpenCV stage completed successfully.")

#--------------------------------------------------------------------------------------
#for Output run this below command in terminal
#python person_1_video_yolo/preprocessing/video_processor.py    
#--------------------------------------------------------------------------------------