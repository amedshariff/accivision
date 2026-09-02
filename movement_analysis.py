import csv
import os
from collections import defaultdict


# ============================================================
# PERSON 2 - MOVEMENT ANALYSIS
# ============================================================

INPUT_FILE = "output/tracking_data.csv"
REPORT_FILE = "output/movement_report.txt"


# ------------------------------------------------------------
# CHECK INPUT FILE
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print("ERROR: tracking_data.csv not found.")
    print("Run tracking.py first.")
    exit()


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

person_data = defaultdict(list)


# ------------------------------------------------------------
# READ TRACKING DATA
# ------------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        person_id = row["Person_ID"]

        person_data[person_id].append({
            "x": float(row["X"]),
            "y": float(row["Y"]),
            "direction": row["Direction"],
            "speed": float(row["Speed_meters_per_second"])
        })


# ------------------------------------------------------------
# CREATE REPORT
# ------------------------------------------------------------

report_lines = []

report_lines.append("==============================================")
report_lines.append("       PERSON 2 - MOVEMENT ANALYSIS")
report_lines.append("==============================================")
report_lines.append("")


# ------------------------------------------------------------
# TOTAL PEOPLE
# ------------------------------------------------------------

total_people = len(person_data)

report_lines.append(
    f"Total unique people tracked: {total_people}"
)

report_lines.append("")


# ------------------------------------------------------------
# ANALYZE EACH PERSON
# ------------------------------------------------------------

for person_id, records in person_data.items():

    report_lines.append("----------------------------------------------")

    report_lines.append(
        f"PERSON {person_id}"
    )

    report_lines.append("----------------------------------------------")


    # --------------------------------------------------------
    # NUMBER OF OBSERVATIONS
    # --------------------------------------------------------

    observations = len(records)

    report_lines.append(
        f"Observations: {observations}"
    )


    # --------------------------------------------------------
    # POSITIONS
    # --------------------------------------------------------

    first_position = (
        records[0]["x"],
        records[0]["y"]
    )

    last_position = (
        records[-1]["x"],
        records[-1]["y"]
    )


    report_lines.append(
        f"Starting position: "
        f"({first_position[0]:.0f}, {first_position[1]:.0f})"
    )

    report_lines.append(
        f"Final position: "
        f"({last_position[0]:.0f}, {last_position[1]:.0f})"
    )


    # --------------------------------------------------------
    # TOTAL MOVEMENT DISTANCE
    # --------------------------------------------------------

    total_distance_pixels = 0

    for i in range(1, len(records)):

        dx = (
            records[i]["x"] -
            records[i - 1]["x"]
        )

        dy = (
            records[i]["y"] -
            records[i - 1]["y"]
        )

        distance = (
            (dx ** 2 + dy ** 2) ** 0.5
        )

        total_distance_pixels += distance


    report_lines.append(
        f"Total movement: "
        f"{total_distance_pixels:.2f} pixels"
    )


    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    speeds = [
        record["speed"]
        for record in records
        if record["speed"] > 0
    ]


    if speeds:

        average_speed = (
            sum(speeds) / len(speeds)
        )

        maximum_speed = max(speeds)

    else:

        average_speed = 0

        maximum_speed = 0


    report_lines.append(
        f"Average speed: "
        f"{average_speed:.2f} m/s"
    )

    report_lines.append(
        f"Maximum speed: "
        f"{maximum_speed:.2f} m/s"
    )


    # --------------------------------------------------------
    # DIRECTION COUNTS
    # --------------------------------------------------------

    direction_count = defaultdict(int)

    for record in records:

        direction = record["direction"]

        direction_count[direction] += 1


    report_lines.append("")

    report_lines.append("Movement directions:")


    for direction, count in direction_count.items():

        report_lines.append(
            f"  {direction}: {count} observations"
        )


    # --------------------------------------------------------
    # MOST COMMON DIRECTION
    # --------------------------------------------------------

    if direction_count:

        dominant_direction = max(
            direction_count,
            key=direction_count.get
        )

    else:

        dominant_direction = "Unknown"


    report_lines.append(
        f"Dominant direction: "
        f"{dominant_direction}"
    )


    # --------------------------------------------------------
    # STATIONARY DETECTION
    # --------------------------------------------------------

    stationary_count = direction_count.get(
        "Stationary",
        0
    )


    moving_count = (
        observations -
        stationary_count
    )


    report_lines.append(
        f"Moving observations: {moving_count}"
    )

    report_lines.append(
        f"Stationary observations: "
        f"{stationary_count}"
    )


    # --------------------------------------------------------
    # MOVEMENT STATUS
    # --------------------------------------------------------

    if moving_count == 0:

        status = "Stationary"

    elif stationary_count > moving_count:

        status = "Mostly Stationary"

    else:

        status = "Moving"


    report_lines.append(
        f"Overall status: {status}"
    )

    report_lines.append("")


# ------------------------------------------------------------
# SAVE REPORT
# ------------------------------------------------------------

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


# ------------------------------------------------------------
# DISPLAY REPORT
# ------------------------------------------------------------

print()

print(
    "\n".join(report_lines)
)

print()

print("==============================================")
print("Movement analysis completed.")
print("==============================================")

print()

print(
    f"Report saved to: {REPORT_FILE}"
)