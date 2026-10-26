import numpy as np
import cv2
import time
from ultralytics import YOLO
import torch
import torchreid
import torchvision.transforms as transforms
from PIL import Image
import csv
from datetime import datetime
import os
import requests
from dotenv import load_dotenv


# ==========================================
# CONFIG
# ==========================================

load_dotenv()

video_path = r"C:\Luminar\Data Science\Project\DeepLearning_Project\multimodel-ai-surveillance\data\videos\fall.mp4"

GLOBAL_LOITERING_THRESHOLD = 30       # anywhere in camera

RESTRICTED_LOITERING_THRESHOLD = 10   # inside restricted 

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError(
        "BOT_TOKEN or CHAT_ID missing in .env"
    )

# ENTRANCE ZONE
ENTRANCE_X1 = 0
ENTRANCE_Y1 = 140 
ENTRANCE_X2 = 220
ENTRANCE_Y2 = 520

# RESTRICTED AREA
RESTRICTED_X1 = 520
RESTRICTED_Y1 = 80
RESTRICTED_X2 = 820
RESTRICTED_Y2 = 420

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO("yolov8s.pt")     # use yolov8m.pt for accurate detection but it is slow

# Load ReID Model

reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

reid_model.eval()

cap = cv2.VideoCapture(video_path)

# cap = cv2.VideoCapture(
#     "http://10.245.29.235:8080/video"
# )

#========================================
print("Camera Opened:", cap.isOpened())
#========================================

if not cap.isOpened():
    print("Cannot open video")
    exit()

print("Starting Loitering Detection System...")

# ==========================================
# RE-ID FEATURE EXTRACTION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
])

def extract_embedding(person_crop):

    image = Image.fromarray(
        cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    )

    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        features = reid_model(image)

    return features.squeeze().numpy()

# cosine_similarity()

def cosine_similarity(vec1, vec2):

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    return np.dot(vec1, vec2)

# Telegram Function

def send_telegram_alert(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, data=data)

    except Exception as e:
        print("Telegram Error:", e)



def send_telegram_photo(photo_path, caption=""):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:

        with open(photo_path, "rb") as photo:

            requests.post(
                url,
                data={
                    "chat_id": CHAT_ID,
                    "caption": caption
                },
                files={
                    "photo": photo
                }
            )

    except Exception as e:

        print("Telegram Photo Error:", e)



# ==========================================
# ALERT LOG FILE
# ==========================================

os.makedirs("data/logs", exist_ok=True)

log_file = "data/logs/alerts.csv"

snapshot_folder = "data/snapshots" 

os.makedirs(snapshot_folder, exist_ok=True)

if not os.path.exists(log_file):

    with open(log_file, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "Person_ID",
            "Master_ID",
            "Event",
            "Snapshot"
        ])

# Store first appearance time
person_entry_time = {}     # whole footage timer

restricted_zone_time = {}    # restricted area timer

logged_events = set()   #Prevent Duplicate Logs

restricted_logged = set()

active_ids = set()

last_seen = {}

previous_positions = {}  # Running speed tracking

running_logged = set()

running_counter = {}

entrance_logged = set()

fall_logged = set() # for fall detection

fall_positions = {}

fall_counter = {}

fall_candidate = {}

person_database = {}  # re id database for storing person embeddings

next_person_id = 1  

track_to_master = {} # Assign Master ID Once Per YOLO Track

REID_THRESHOLD = 0.75 #(0.75 standard)

RUNNING_SPEED_THRESHOLD = 80  # Adjust this value based on your video resolution and frame rate

FALL_RATIO_THRESHOLD = 1.5  # Adjust this value based on your video resolution and frame rate

FALL_DROP_THRESHOLD = 15

FALL_CONFIRM_FRAMES = 3  # Number of consecutive frames to confirm a fall (3 frames)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.3,     # 0.5 normal
        imgsz=960,
        verbose=False
    )

    annotated_frame = frame.copy()

    # ==========================================
    # ALERT PANEL
    # ==========================================

    alert_y = 40

    # Draw Restricted Zone

    # ENTRANCE ZONE

    cv2.rectangle(
        annotated_frame,
        (ENTRANCE_X1, ENTRANCE_Y1),
        (ENTRANCE_X2, ENTRANCE_Y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        "ENTRANCE ZONE",
        (ENTRANCE_X1 + 10, ENTRANCE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # RESTRICTED AREA

    cv2.rectangle(
        annotated_frame,
        (RESTRICTED_X1, RESTRICTED_Y1),
        (RESTRICTED_X2, RESTRICTED_Y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "RESTRICTED AREA",
        (RESTRICTED_X1 + 10, RESTRICTED_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box in boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            active_ids.add(track_id)

            last_seen[track_id] = time.time()

            # ==========================================
            # GLOBAL TIMER
            # ==========================================

            if track_id not in person_entry_time:
                person_entry_time[track_id] = time.time()

            global_duration = int(
                time.time() - person_entry_time[track_id]
            )

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Calculate aspect ratio for fall detection(w&h)

            width = x2 - x1
            height = y2 - y1

            aspect_ratio = width / height


            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # ==========================================
            # PERSON CROP FOR RE-ID
            # ==========================================

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # ==========================================
            # EXTRACT EMBEDDING
            # ==========================================

            if track_id in track_to_master:

                master_id = track_to_master.get(track_id, track_id)

            else:

                embedding = extract_embedding(person_crop)

                best_match = None
                best_score = 0

                for person_id, stored_embedding in person_database.items():

                    score = cosine_similarity(
                        embedding,
                        stored_embedding
                    )

                    if score > best_score:

                        best_score = score
                        best_match = person_id
                    #====================================
                    # print(
                    #     f"TRACK:{track_id} "
                    #     f"BEST_MATCH:{best_match} "
                    #     f"SCORE:{best_score:.2f}"
                    # )
                    #=====================================
                
                
                if best_score > REID_THRESHOLD:

                    master_id = best_match

                    # Update stored embedding
                    person_database[master_id] = (
                        0.8 * person_database[master_id]
                        + 0.2 * embedding
                    )

                else:

                    master_id = next_person_id

                    person_database[master_id] = embedding

                    next_person_id += 1

                track_to_master[track_id] = master_id
                #=========================================
                # print(
                #     f"NEW TRACK {track_id} "
                #     f"MATCHED TO MASTER {master_id}"
                # )
                #=========================================


            previous_center_y = center_y

            if track_id in fall_positions:
                previous_center_y = fall_positions[track_id]

            fall_positions[track_id] = center_y


            # Calculate running speed

            speed = 0

            if track_id in previous_positions:

                prev_x, prev_y = previous_positions[track_id]

                speed = ((center_x - prev_x) ** 2 +
                        (center_y - prev_y) ** 2) ** 0.5
                
            #============================================

            # print(f"ID:{track_id} Speed:{speed:.2f}")

            #============================================

            previous_positions[track_id] = (
                center_x,
                center_y
            )

            # Fall calculation

            vertical_drop = center_y - previous_center_y

            horizontal_body = (
                aspect_ratio > FALL_RATIO_THRESHOLD
            )

            sudden_drop = (
                vertical_drop > FALL_DROP_THRESHOLD
            )

            #====================================

            # print(
            #     f"ID:{track_id} "
            #     f"Drop:{vertical_drop} "
            #     f"AR:{aspect_ratio:.2f}"
            #     f"Horizontal:{horizontal_body} "
            #     f"Sudden:{sudden_drop}"
            # )

            #====================================

            # Default values

            duration = 0
            color = (0, 255, 0)

            inside_restricted = (
                RESTRICTED_X1 < center_x < RESTRICTED_X2
                and
                RESTRICTED_Y1 < center_y < RESTRICTED_Y2
            )

            # Person entered restricted area

            if inside_restricted:
  
                if track_id not in restricted_zone_time:
                    restricted_zone_time[track_id] = time.time()

                duration = int(
                    time.time() - restricted_zone_time[track_id]
                )

                event_key = (track_id, "Restricted")   #Save Restricted Area Alerts

                if event_key not in restricted_logged:
                    

                    with open(log_file, "a", newline="") as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            track_id,
                            master_id,
                            "Restricted Area"
                        ])

                    restricted_logged.add(event_key)

                    send_telegram_alert(
                        f"⛔ RESTRICTED AREA ALERT\n"
                        f"Master ID: {master_id}\n"
                        f"Track ID: {track_id}\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                # Loitering Alert

                if duration >= RESTRICTED_LOITERING_THRESHOLD:

                    color = (255, 0, 255)


                # ==========================================
                # SAVE LOITERING EVENT TO CSV
                # ==========================================

                    event_key = (track_id, "High Risk")

                    if event_key not in logged_events:

                        with open(log_file, "a", newline="") as file:

                            writer = csv.writer(file)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            snapshot_path = os.path.join(
                                snapshot_folder,
                                f"master_{master_id}_highrisk_{timestamp}.jpg"
                            )

                            cv2.imwrite(snapshot_path, frame)

                            send_telegram_photo(
                                snapshot_path,
                                f"🚨 HIGH RISK ALERT\n"
                                f"Master ID: {master_id}\n"
                                f"Track ID: {track_id}\n"
                                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            )

                            writer.writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                track_id,
                                master_id,
                                "High Risk",
                                snapshot_path
                            ])


                        logged_events.add(event_key)

                # ==========================================
                # SHOW ALERT ON SCREEN
                # ==========================================


                    cv2.putText(
                        annotated_frame,
                        f"HIGH RISK ALERT : ID {track_id}",
                        (20, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

                    alert_y += 30

            # Person left restricted area

            else:

                if track_id in restricted_zone_time:
                    del restricted_zone_time[track_id]

 

                logged_events.discard(
                    (track_id, "High Risk")
                )
                        

            # ENTRANCE DETECTION

            if (
                ENTRANCE_X1 < center_x < ENTRANCE_X2
                and
                ENTRANCE_Y1 < center_y < ENTRANCE_Y2
            ):
                
                event_key = (track_id, "Entrance")

                if event_key not in entrance_logged:

                    with open(log_file, "a", newline="") as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            track_id,
                            master_id,
                            "Entrance"
                        ])

                    entrance_logged.add(event_key)

                cv2.putText(
                    annotated_frame,
                    f"ENTRY DETECTED ID:{track_id}",
                    (x1, y1 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )


            # -------------------------
            # Restricted Area Alert
            # -------------------------
            
            # Only set RED if person is NOT loitering yet
            if inside_restricted and duration < RESTRICTED_LOITERING_THRESHOLD:
                color = (0, 0, 255)

                cv2.putText(
                    annotated_frame,
                    f"RESTRICTED AREA ALERT ID:{track_id}",
                    (20, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                alert_y += 30

            #=================================================================
            # Running Detection Counter
            #=================================================================

            if speed > RUNNING_SPEED_THRESHOLD:

                running_counter[track_id] = (
                    running_counter.get(track_id, 0) + 1
                )

            else:

                running_counter[track_id] = 0

            
            # Running Alert after 3 consecutive frames

            if running_counter.get(track_id, 0) >= 3:

                cv2.putText(
                    annotated_frame,
                    f"RUNNING DETECTED ID:{track_id}",
                    (20, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                alert_y += 30

                print(
                    f"RUNNING ALERT -> "
                    f"ID:{track_id} "
                    f"Speed:{speed:.2f}"
                )

                event_key = (track_id, master_id, "Running")

                if event_key not in running_logged:

                    with open(log_file, "a", newline="") as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            track_id,
                            master_id,
                            "Running"
                        ])

                    running_logged.add(event_key)

                    send_telegram_alert(
                        f"🏃 RUNNING DETECTED\n"
                        f"Master ID: {master_id}\n"
                        f"Track ID: {track_id}\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )


            # ==========================================
            # SMART FALL DETECTION
            # ==========================================


            # Stage 1: Sudden drop detected
            if sudden_drop:

                fall_candidate[track_id] = 30   # keep candidate alive for 30 frames


            # Stage 2: Countdown
            if track_id in fall_candidate:

                fall_candidate[track_id] -= 1

                if fall_candidate[track_id] <= 0:
                    del fall_candidate[track_id]


            # Stage 3: Horizontal posture after drop
            if (
                track_id in fall_candidate
                and horizontal_body
            ):

                fall_counter[track_id] = (
                    fall_counter.get(track_id, 0) + 1
                )

            else:

                fall_counter[track_id] = 0
            
            if fall_counter.get(track_id, 0) >= FALL_CONFIRM_FRAMES:

                #=======================================
                # print("FALL DETECTED")
                #=======================================

                cv2.putText(
                    annotated_frame,
                    f"FALL DETECTED ID:{track_id}",
                    (20, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                alert_y += 30

                event_key = (track_id, master_id, "Fall")

                if event_key not in fall_logged:

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    snapshot_path = os.path.join(
                        snapshot_folder,
                        f"master_{master_id}_fall_{timestamp}.jpg"
                    )

                    cv2.imwrite(snapshot_path, frame)

                    send_telegram_photo(
                        snapshot_path,
                        f"🚨 FALL DETECTED\n"
                        f"Master ID: {master_id}\n"
                        f"Track ID: {track_id}\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    with open(log_file, "a", newline="") as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            track_id,
                            master_id,
                            "Fall Detected",
                            snapshot_path
                        ])

                    fall_logged.add(event_key)


            # ==========================================
            # GLOBAL LOITERING ALERT
            # ==========================================

            if global_duration >= GLOBAL_LOITERING_THRESHOLD:

                # Don't overwrite High Risk purple color

                if color != (255, 0, 255):
                    color = (0, 165, 255)

                # Save to CSV only once

                event_key = (track_id, master_id, "Global Loitering")

                if event_key not in logged_events:

                    with open(log_file, "a", newline="") as file:

                        writer = csv.writer(file)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        snapshot_path = os.path.join(
                            snapshot_folder,
                            f"master_{master_id}_loitering_{timestamp}.jpg"
                        )

                        cv2.imwrite(snapshot_path, frame)

                        send_telegram_photo(
                            snapshot_path,
                            f"⚠️ GLOBAL LOITERING\n"
                            f"Master ID: {master_id}\n"
                            f"Track ID: {track_id}\n"
                            f"Duration: {global_duration}s"
                        )

                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            track_id,
                            master_id,
                            "Global Loitering",
                            snapshot_path
                        ])

                    logged_events.add(event_key)

                # Show alert on screen

                cv2.putText(
                    annotated_frame,
                    f"LOITERING ALERT ID:{track_id}",
                    (20, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                alert_y += 30

            # -------------------------
            # Draw Person Box
            # -------------------------

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                annotated_frame,
                f"ID:{track_id} | {global_duration}s | SPD:{int(speed)}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # ==========================================
    # CLEAN UP IDs THAT LEFT THE SCENE
    # ==========================================

    for old_id in list(person_entry_time.keys()):

        if (
            old_id not in active_ids
            and
            time.time() - last_seen.get(old_id, 0) > 5
        ):   

            del person_entry_time[old_id]


            if old_id in restricted_zone_time:
                del restricted_zone_time[old_id]

          


            restricted_logged.discard((old_id, "Restricted"))  


            logged_events.discard((old_id, "High Risk"))

            logged_events.discard((old_id, "Global Loitering"))

            previous_positions.pop(old_id, None)

            entrance_logged.discard((old_id, "Entrance"))

            master = track_to_master.get(old_id)

            running_logged.discard((old_id, master, "Running"))

            fall_logged.discard((old_id, master, "Fall"))
            
            running_counter.pop(old_id, None)

            fall_positions.pop(old_id, None)

            fall_candidate.pop(old_id, None)

            fall_counter.pop(old_id, None)


            if old_id in last_seen:
                del last_seen[old_id]

            if old_id in track_to_master:
                del track_to_master[old_id]

    # Reset for next frame

    active_ids.clear()

    cv2.imshow(
        "AI Surveillance - Loitering Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()