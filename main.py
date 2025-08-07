import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import time
import numpy as np
import winsound
import tkinter as tk
from PIL import Image, ImageTk

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize FaceMesh detector from CVZone
detector = FaceMeshDetector(maxFaces=1)

# Adjusted thresholds for glasses wearers (lower EAR threshold, higher MAR threshold)
EAR_THRESHOLD = 0.18  # Lowered threshold for glasses holders
EAR_CONSEC_FRAMES = 35  # Slightly more frames to avoid false positives
MAR_THRESHOLD = 0.65   # Slightly higher for yawn
MAR_CONSEC_FRAMES = 15

# Head pose thresholds (not used, head down detection disabled)
# HEAD_DOWN_PITCH_THRESHOLD = 18  # degrees, adjust as needed

# Initialize counters and timers
eye_closed_counter = 0
yawn_counter = 0
drowsy = False
yawning = False
eye_close_start_time = None  # For timing how long eyes are closed
eye_close_duration = 0
alarm_active = False  # Track if alarm is currently sounding

# Indices for left and right eyes and mouth landmarks (CVZone FaceMesh indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 308, 311, 402, 14, 178, 87, 317]
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

def euclidean_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def eye_aspect_ratio(eye_points):
    A = euclidean_dist(eye_points[1], eye_points[5])
    B = euclidean_dist(eye_points[2], eye_points[4])
    C = euclidean_dist(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_points):
    A = euclidean_dist(mouth_points[1], mouth_points[7])  # 81-178
    B = euclidean_dist(mouth_points[2], mouth_points[6])  # 13-14
    C = euclidean_dist(mouth_points[3], mouth_points[5])  # 308-402
    D = euclidean_dist(mouth_points[0], mouth_points[4])  # 78-311 (horizontal)
    mar = (A + B + C) / (3.0 * D)
    return mar

def get_head_pose(face):
    # 3D model points of facial landmarks (approximate, for pose estimation)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -63.6, -12.5),         # Chin
        (-43.3, 32.7, -26.0),        # Left eye left corner
        (43.3, 32.7, -26.0),         # Right eye right corner
        (-28.9, -28.9, -24.1),       # Left Mouth corner
        (28.9, -28.9, -24.1)         # Right mouth corner
    ])
    # 2D image points from detected face
    image_points = np.array([
        face[NOSE_TIP],          # Nose tip
        face[CHIN],              # Chin
        face[LEFT_EYE_CORNER],   # Left eye left corner
        face[RIGHT_EYE_CORNER],  # Right eye right corner
        face[LEFT_MOUTH_CORNER], # Left Mouth corner
        face[RIGHT_MOUTH_CORNER] # Right mouth corner
    ], dtype="double")
    # Camera internals
    size = (800, 600)
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    # SolvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0, 0, 0
    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [float(angle) for angle in euler_angles]
    return pitch, yaw, roll

# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("Drowsiness & Yawn Detection")
root.configure(bg="#222831")

window_width = 900
window_height = 700
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

title_label = tk.Label(root, text="Drowsiness & Yawn Detection", font=("Arial", 24, "bold"), bg="#222831", fg="#00adb5")
title_label.pack(pady=10)

video_frame = tk.Label(root, bg="#393e46")
video_frame.pack(pady=10)

info_frame = tk.Frame(root, bg="#222831")
info_frame.pack(pady=10)

ear_label = tk.Label(info_frame, text="EAR: --", font=("Arial", 16), bg="#222831", fg="#eeeeee")
ear_label.grid(row=0, column=0, padx=20)

mar_label = tk.Label(info_frame, text="MAR: --", font=("Arial", 16), bg="#222831", fg="#eeeeee")
mar_label.grid(row=0, column=1, padx=20)

alert_label = tk.Label(root, text="", font=("Arial", 20, "bold"), bg="#222831")
alert_label.pack(pady=10)

eye_time_label = tk.Label(root, text="", font=("Arial", 16), bg="#222831", fg="#00adb5")
eye_time_label.pack(pady=5)

def update_frame():
    global eye_closed_counter, yawn_counter, drowsy, yawning
    global eye_close_start_time, eye_close_duration
    global alarm_active

    success, img = cap.read()
    if not success:
        root.after(10, update_frame)
        return

    img = cv2.flip(img, 1)

    img, faces = detector.findFaceMesh(img, draw=False)

    avg_ear = 0
    mar = 0
    drowsy = False
    yawning = False
    alert_text = ""
    alert_color = "#222831"
    eye_close_duration = 0

    if faces:
        face = faces[0]

        left_eye = [face[i] for i in LEFT_EYE]
        right_eye = [face[i] for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        mouth = [face[i] for i in MOUTH]
        mar = mouth_aspect_ratio(mouth)

        # Drowsiness detection based on EAR (alarm after 1s, alarm stays on until eyes open)
        if avg_ear < EAR_THRESHOLD:
            if eye_closed_counter == 0:
                eye_close_start_time = time.time()
            eye_closed_counter += 1
            if eye_close_start_time is not None:
                eye_close_duration = time.time() - eye_close_start_time
            # Mark if eyes have been closed for at least 1 second
            if eye_close_duration >= 1:
                drowsy = True
                alert_text = "DROWSINESS ALERT!"
                alert_color = "#ff2e63"
                if not alarm_active:
                    alarm_active = True
            else:
                drowsy = False
                alert_text = ""
                alert_color = "#222831"
        else:
            # Eyes are open now
            eye_closed_counter = 0
            eye_close_start_time = None
            eye_close_duration = 0
            alarm_active = False

        # Yawn detection based on MAR
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= MAR_CONSEC_FRAMES:
                yawning = True
                if not drowsy:
                    alert_text = "YAWN ALERT!"
                    alert_color = "#f9ed69"
                winsound.Beep(2000, 100)
        else:
            yawn_counter = 0
            yawning = False

        # Draw overlays on the frame
        cv2.rectangle(img, (10, 10), (340, 90), (34, 40, 49), -1)
        cv2.putText(img, f'EAR: {avg_ear:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 173, 181), 2)
        cv2.putText(img, f'MAR: {mar:.2f}', (170, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 173, 181), 2)
        if eye_close_duration > 0:
            cv2.putText(img, f'Eye Closed: {eye_close_duration:.1f}s', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Drowsiness alert overlay (if drowsy)
        if drowsy:
            cv2.rectangle(img, (10, 90), (400, 140), (46, 204, 113), -1)
            cv2.putText(img, 'DROWSINESS ALERT!', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 46, 99), 2)
        elif yawning:
            cv2.rectangle(img, (10, 90), (400, 140), (255, 237, 105), -1)
            cv2.putText(img, 'YAWN ALERT!', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 46, 99), 3)

    # Update Tkinter labels
    ear_label.config(text=f"EAR: {avg_ear:.2f}" if avg_ear else "EAR: --")
    mar_label.config(text=f"MAR: {mar:.2f}" if mar else "MAR: --")
    alert_label.config(text=alert_text, fg=alert_color)
    if eye_close_duration > 0:
        eye_time_label.config(text=f"Eyes closed: {eye_close_duration:.1f} seconds")
    else:
        eye_time_label.config(text="")

    # Sound alarm if needed (alarm stays on as long as eyes are closed for >= 1s)
    if alarm_active:
        # Play a short beep every frame (approx every 10ms, so it will sound continuous)
        winsound.Beep(2000, 100)

    # Convert the image to RGB and then to ImageTk
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((800, 600))
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_frame.imgtk = imgtk
    video_frame.configure(image=imgtk)

    root.after(10, update_frame)

def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
root.mainloop()
