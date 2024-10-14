import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load COCO class list
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define parking areas
areas = {
    1: [(42, 364), (20, 425), (76, 428), (90, 364)],
    2: [(105, 353), (86, 428), (137, 427), (146, 358)],
    3: [(159, 354), (150, 427), (204, 425), (203, 353)],
    4: [(217, 352), (219, 422), (273, 418), (261, 347)],
    5: [(274, 345), (286, 417), (338, 415), (321, 345)],
    6: [(336, 343), (357, 410), (409, 408), (382, 340)],
    7: [(396, 338), (426, 404), (479, 399), (439, 334)],
    8: [(458, 333), (494, 397), (543, 390), (495, 330)],
    9: [(511, 327), (557, 388), (603, 383), (549, 324)],
    10: [(564, 323), (615, 381), (654, 372), (596, 315)],
    11: [(616, 316), (666, 369), (703, 363), (642, 312)],
    12: [(674, 311), (730, 360), (764, 355), (707, 308)],
}

# Initialize video capture
cap = cv2.VideoCapture('parking1.mp4')

def draw_parking_area(frame, area_id, is_occupied):
    color = (0, 0, 255) if is_occupied else (0, 255, 0)
    cv2.polylines(frame, [np.array(areas[area_id], np.int32)], True, color, 2)
    cv2.putText(frame, str(area_id), areas[area_id][0], cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

def detect_parking(frame, results, areas):
    parking_status = {i: 0 for i in areas.keys()}
    # Iterate over detected objects
    for box in results[0].boxes.data:
        x1, y1, x2, y2, _, class_id = map(int, box)
        label = class_list[class_id]

        # Check if it's a car
        if 'car' in label:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check in which area the car is located
            for area_id, points in areas.items():
                if cv2.pointPolygonTest(np.array(points, np.int32), (cx, cy), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    parking_status[area_id] = 1
                    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # Draw areas with occupancy information
    free_spaces = 0
    for area_id, occupied in parking_status.items():
        draw_parking_area(frame, area_id, occupied)
        if not occupied:
            free_spaces += 1

    # Display available spaces
    cv2.putText(frame, f"Free spaces: {free_spaces}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return free_spaces

# Main loop for video processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (1020, 500))

    # YOLOv8 detection
    results = model.predict(frame, stream=False)

    # Detect parking and display available spaces
    detect_parking(frame, results, areas)

    # Display video
    cv2.imshow('Parking Monitor', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
