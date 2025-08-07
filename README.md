# 💤 Drowsiness & Yawn Detection System

![Preview](preview.gif)

This project is a real-time drowsiness and yawn detection system using **OpenCV**, **CVZone**, and **Tkinter GUI**. It uses **facial landmark detection** to monitor eye and mouth aspect ratios to detect signs of fatigue or yawning.

---

## ▶️ How to Run

1. Clone or download this repository:
   ```bash
   git clone https://github.com/your-username/drowsiness-detector.git
   cd drowsiness-detector
2.Install the required packages:
```
pip install opencv-python
pip install numpy
pip install cvzone
pip install pillow

```
3.Run the application:
```
python main.py

```
# 🖥️ GUI Interface
📷 Live webcam feed in a Tkinter window

👁️ Displays EAR (Eye Aspect Ratio) in real time

👄 Displays MAR (Mouth Aspect Ratio) in real time

⏱️ Tracks how long the eyes are closed

🚨 Visual and audio alerts for drowsiness and yawning

🎨 Clean interface with modern styling

# 📌 Notes
EAR threshold is set to 0.18 (for people with glasses)

MAR threshold is 0.65 (to avoid false yawns)

Works best with:

Good lighting

Webcam at eye level

Single face in the frame

You can adjust the sensitivity by modifying:
```
EAR_THRESHOLD = 0.18
MAR_THRESHOLD = 0.65
```
📁 drowsiness-detector
├── main.py
├── preview.gif
└── README.md


# 👨‍💻 Author
Made with ❤️ by [Mejri Neder]
Feel free to contribute or suggest improvements!
