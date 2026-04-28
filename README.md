# Gesture Virtual Mouse v2

A touchless computer mouse interface that uses hand gestures captured by a webcam. Built with MediaPipe, OpenCV, and PyAutoGUI.

## 🚀 Features

- **Smooth Cursor Control**: Move the cursor by lifting your index finger (keep thumb folded to move).
- **Intelligent Left Click**: Pinch thumb and index together to click.
- **Double Click Support**: Perform two quick pinches within 0.4s to double-click.
- **Drag & Drop**: Hold a pinch for 0.6s to start dragging; release to drop.
- **Natural Right Click**: Hold a "V" sign (index and middle fingers up, thumb folded) for 0.7s.
- **High-Speed Scrolling**: Hold three fingers up and move your hand up/down. Features a float-accumulator for high sensitivity.
- **Activation Toggle**: Hold an open palm for 1 second to enable/disable gesture control.
- **Console Controls**: Press `Q` in the terminal to quit the application safely.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd gesture_mouse
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe pyautogui numpy
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## 🎮 Gesture Guide

| Gesture | Action |
| :--- | :--- |
| ☝️ **Index Up + Thumb Folded** | Move Cursor |
| 🤏 **Quick Pinch** | Left Click |
| 🤏🤏 **Two Quick Pinches** | Double Click |
| 🤏 **Hold Pinch (0.6s)** | Drag & Drop |
| ✌️ **V-Sign (Thumb Folded)** | Right Click |
| 🤟 **3 Fingers + Hand Move** | Scroll |
| 🖐️ **Open Palm (1s)** | Toggle ON/OFF |

## ⚙️ Configuration

You can tweak sensitivity and thresholds at the top of `main.py`:
- `SMOOTHING`: Adjust cursor lag/smoothness.
- `SCROLL_SENS`: Adjust scroll speed.
- `PINCH_THRESHOLD`: Adjust how close fingers must be to click.

## ⚖️ License
MIT
