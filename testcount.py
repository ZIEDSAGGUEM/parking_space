import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import csv
from datetime import datetime
import argparse
import threading
import sqlite3
import smtplib
from email.mime.text import MIMEText
import matplotlib
matplotlib.use('Agg')  # use non-GUI backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Flask, jsonify, render_template, request

# ------------------------------------------------------
# Global Data Structure (shared across threads)
# ------------------------------------------------------
latest_data = {
    'timestamp': None,
    'frame_no': 0,
    'occupancy': {},       # {area_id: True/False}
    'free_spaces': None,
    'history': []          # list of free spaces counts per frame
}

# ------------------------------------------------------
# Database Setup (SQLite)
# ------------------------------------------------------
db_conn = sqlite3.connect("parking_data.db", check_same_thread=False)
db_cursor = db_conn.cursor()
db_cursor.execute("""
    CREATE TABLE IF NOT EXISTS occupancy (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        frame INTEGER,
        free_spaces INTEGER
    )
""")
db_conn.commit()

def log_to_db(timestamp, frame_no, free_spaces):
    db_cursor.execute("INSERT INTO occupancy (timestamp, frame, free_spaces) VALUES (?, ?, ?)",
                      (timestamp, frame_no, free_spaces))
    db_conn.commit()

# ------------------------------------------------------
# Optional Email Alert Function (requires configuration)
# ------------------------------------------------------
def send_alert_email(subject, message):
    # Configure these with your email settings
    sender = "talksquadweb@gmail.com"
    receiver = "ziedsagguem@gmail.com"
    password = "ZIED1234"
    
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("Alert email sent.")
    except Exception as e:
        print("Failed to send email alert:", e)

# ------------------------------------------------------
# Dashboard Generation: Create a Matplotlib chart image
# ------------------------------------------------------
def get_history_chart_image(history, width=300, height=200):
    fig = Figure(figsize=(width/100, height/100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(history, "b-", marker="o")
    ax.set_title("Free Spaces History")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Free Spaces")
    fig.tight_layout()
    canvas.draw()
    chart_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    chart_image = chart_image.reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 3))
    chart_image = cv2.cvtColor(chart_image, cv2.COLOR_RGB2BGR)
    chart_image = cv2.resize(chart_image, (width, height))
    return chart_image

def generate_dashboard_image(latest_data, parking_areas, dashboard_width=300, dashboard_height=500):
    dashboard_img = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
    text_area_height = dashboard_height // 2
    chart_area_height = dashboard_height - text_area_height

    # Add occupancy info text
    y_offset = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    for area in parking_areas:
        occ = latest_data['occupancy'].get(area['id'], False)
        status_text = f"Area {area['id']}: {'Occupied' if occ else 'Free'}"
        cv2.putText(dashboard_img, status_text, (10, y_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_offset += 20
    free_text = f"Free Spaces: {latest_data['free_spaces']}"
    cv2.putText(dashboard_img, free_text, (10, y_offset), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # Add the history chart in the bottom half
    if latest_data['history']:
        chart_img = get_history_chart_image(latest_data['history'], width=dashboard_width, height=chart_area_height)
    else:
        chart_img = np.zeros((chart_area_height, dashboard_width, 3), dtype=np.uint8)
    dashboard_img[text_area_height:dashboard_height, 0:dashboard_width] = chart_img
    return dashboard_img

# ------------------------------------------------------
# Flask Web Dashboard (runs in a separate thread)
# ------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    # For a production app, create a proper HTML template.
    html = "<h1>Parking Occupancy Dashboard</h1>"
    html += f"<p>Timestamp: {latest_data['timestamp']}</p>"
    html += f"<p>Frame: {latest_data['frame_no']}</p>"
    html += f"<p>Free Spaces: {latest_data['free_spaces']}</p>"
    html += "<ul>"
    for area, status in latest_data['occupancy'].items():
        html += f"<li>Area {area}: {'Occupied' if status else 'Free'}</li>"
    html += "</ul>"
    return html

@app.route('/data')
def data():
    return jsonify(latest_data)

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# ------------------------------------------------------
# Main Video Processing Function
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Enhanced Parking Occupancy Detection with Dashboard, DB Logging & Alerts")
    parser.add_argument("--input", type=str, default="parking1.mp4", help="Input video file path")
    parser.add_argument("--output", type=str, default="output_video.avi", help="Output video file path")
    parser.add_argument("--log", type=str, default="parking_log.csv", help="CSV log file (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = parser.parse_args()

    model = YOLO('yolov8s.pt')
    cap = cv2.VideoCapture(args.input)
    frame_width, frame_height = 1020, 500
    dashboard_width = 300
    combined_width = frame_width + dashboard_width

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (combined_width, frame_height))

    # Load class names (expects coco.txt to exist)
    with open("coco.txt", "r") as f:
        class_list = f.read().strip().split("\n")

    # Define parking areas as a list of dictionaries
    parking_areas = [
        {"id": 1, "points": [(52, 364), (30, 417), (73, 412), (88, 369)]},
        {"id": 2, "points": [(105, 353), (86, 428), (137, 427), (146, 358)]},
        {"id": 3, "points": [(159, 354), (150, 427), (204, 425), (203, 353)]},
        {"id": 4, "points": [(217, 352), (219, 422), (273, 418), (261, 347)]},
        {"id": 5, "points": [(274, 345), (286, 417), (338, 415), (321, 345)]},
        {"id": 6, "points": [(336, 343), (357, 410), (409, 408), (382, 340)]},
        {"id": 7, "points": [(396, 338), (426, 404), (479, 399), (439, 334)]},
        {"id": 8, "points": [(458, 333), (494, 397), (543, 390), (495, 330)]},
        {"id": 9, "points": [(511, 327), (557, 388), (603, 383), (549, 324)]},
        {"id": 10, "points": [(564, 323), (615, 381), (654, 372), (596, 315)]},
        {"id": 11, "points": [(616, 316), (666, 369), (703, 363), (642, 312)]},
        {"id": 12, "points": [(674, 311), (730, 360), (764, 355), (707, 308)]}
    ]

    # Optional: CSV logging (in addition to DB logging)
    csv_file = open(args.log, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["timestamp", "frame"]
    for area in parking_areas:
        header.append(f"area_{area['id']}")
    header.append("free_spaces")
    csv_writer.writerow(header)

    def log_frame_csv(timestamp, frame_no, occupancy, free_spaces):
        row = [timestamp, frame_no]
        for area in parking_areas:
            row.append(1 if occupancy[area["id"]] else 0)
        row.append(free_spaces)
        csv_writer.writerow(row)

    def draw_parking_area(frame, area, occupied):
        pts = np.array(area["points"], np.int32).reshape((-1, 1, 2))
        color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.polylines(frame, [pts], True, color, 2)
        x_vals = [pt[0] for pt in area["points"]]
        y_vals = [pt[1] for pt in area["points"]]
        label_pos = (min(x_vals), min(y_vals)-5)
        cv2.putText(frame, str(area["id"]), label_pos, cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 1)

    previous_free_spaces = None
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        frame = cv2.resize(frame, (frame_width, frame_height))
        results = model.predict(frame)
        detections = results[0].boxes.data.cpu().numpy()

        # Initialize occupancy for each parking area
        occupancy = {area["id"]: False for area in parking_areas}

        for detection in detections:
            x1, y1, x2, y2, score, class_idx = detection
            if score < 0.5:
                continue
            class_idx = int(class_idx)
            label = class_list[class_idx]
            if "car" not in label.lower():
                continue
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            for area in parking_areas:
                pts = np.array(area["points"], np.int32)
                result = cv2.pointPolygonTest(pts, (cx, cy), False)
                if result >= 0:
                    occupancy[area["id"]] = True
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)
                    cv2.putText(frame, label, (int(x1), int(y1)-5),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                    if args.debug:
                        print(f"Detection {label} in area {area['id']} at {(cx, cy)}")

        occupied_count = sum(1 for occ in occupancy.values() if occ)
        free_spaces = len(parking_areas) - occupied_count

        # Draw parking area overlays
        for area in parking_areas:
            draw_parking_area(frame, area, occupancy[area["id"]])
        cv2.putText(frame, f"Free Spaces: {free_spaces}", (20, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        cv2.putText(frame, f"Frame: {frame_no}", (20, 60),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        latest_data['occupancy'] = occupancy
        latest_data['free_spaces'] = free_spaces
        latest_data['frame_no'] = frame_no
        latest_data['timestamp'] = current_time
        latest_data['history'].append(free_spaces)

        # Log to CSV and DB
        log_frame_csv(current_time, frame_no, occupancy, free_spaces)
        log_to_db(current_time, frame_no, free_spaces)

        # Optional: Send email alert if free spaces change dramatically
        if previous_free_spaces is not None and free_spaces > previous_free_spaces:
            # Uncomment and configure to enable alerts
            # send_alert_email("Parking Spot Available", f"Free spaces increased from {previous_free_spaces} to {free_spaces} at {current_time}.")
            pass
        previous_free_spaces = free_spaces

        # Generate dashboard image and combine with main frame
        dashboard_img = generate_dashboard_image(latest_data, parking_areas,
                                                  dashboard_width=dashboard_width, dashboard_height=frame_height)
        combined_frame = cv2.hconcat([frame, dashboard_img])
        out.write(combined_frame)
        cv2.imshow("Parking Occupancy with Dashboard", combined_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc to exit
            break

    cap.release()
    out.release()
    csv_file.close()
    db_conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start Flask web server in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    main()
