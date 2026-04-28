"""
Gesture-Based Virtual Mouse  v2
================================
Controls the computer mouse using natural, easy hand gestures.

GESTURE MAP:
  index finger only          -> Move cursor
  Thumb + Index pinch (tap)  -> Left Click
  Thumb + Index pinch x2     -> Double Click  (two quick taps)
  Thumb + Index pinch (hold) -> Drag & Drop  (hold 0.6 s)
  V sign (hold 0.7 s)        -> Right Click
  3 fingers up + move hand   -> Scroll  (move hand up / down)
  🖐  Open palm held 1 s         -> Toggle ON / OFF

TIPS:
  • Keep your hand 30-60 cm from the webcam
  • Good lighting = better tracking
  • Press Q or Esc to quit the preview window

Safety: move mouse to any screen corner to emergency-stop (PyAutoGUI failsafe)
"""

import os
import sys
import threading
import msvcrt

# ── Silence MediaPipe noise & prevent hang-on-exit ────────────────────────────
os.environ["GLOG_minloglevel"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ.setdefault("MEDIAPIPE_TELEMETRY_ENABLED", "0")

import cv2
import pyautogui
import time
import math
import numpy as np
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image, ImageFormat

# ─── Hand skeleton connection pairs ───────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — tweak these to suit your hand size and lighting
# ═══════════════════════════════════════════════════════════════════════════════
CAMERA_ID        = 0
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480

# Cursor
PADDING          = 80      # px from frame edge → cursor can still reach screen border
SMOOTHING        = 6       # higher = smoother but slower  (3–12)

# Left-click / Drag
PINCH_THRESHOLD  = 55      # px – bigger = easier to trigger (try 50–70)
DRAG_HOLD        = 0.6     # seconds pinch must be held to become a drag
CLICK_DEBOUNCE   = 0.18    # min seconds between two left-clicks (tighter for dbl-click)

# Right-click  (V-gesture hold)
RIGHT_CLICK_HOLD = 0.7     # seconds V must be held before right-click fires
RIGHT_DEBOUNCE   = 0.8     # min seconds between two right-clicks

# Scroll  (3-finger, delta-based)
SCROLL_SENS      = 1.5     # hand-pixel movement -> scroll lines (float accumulator)
SCROLL_DEAD_ZONE = 3       # px of hand movement to ignore (prevents drift)

# Double-click
DBLCLICK_WINDOW  = 0.35    # max seconds between two taps to count as double-click

# Toggle
TOGGLE_HOLD      = 1.0     # seconds open palm must be held

# ═══════════════════════════════════════════════════════════════════════════════

# ─── PyAutoGUI ────────────────────────────────────────────────────────────────
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0
screen_w, screen_h = pyautogui.size()

# ─── MediaPipe model ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand landmark model (~5 MB)…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Model ready.")

# ─── MediaPipe landmarker (async / LIVE_STREAM) ───────────────────────────────
_latest_result = None

def _cb(result, _img, _ts):
    global _latest_result
    _latest_result = result

landmarker = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options   = mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode   = mp_vision.RunningMode.LIVE_STREAM,
        num_hands      = 1,
        min_hand_detection_confidence = 0.65,
        min_hand_presence_confidence  = 0.65,
        min_tracking_confidence       = 0.65,
        result_callback = _cb,
    )
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def px(lms, i, w, h):
    l = lms[i]
    return int(l.x * w), int(l.y * h)

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def fingers_up(lms):
    """
    Returns [thumb, index, middle, ring, pinky] bool list.
    Works for both hands (detects orientation from palm landmarks).
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    # thumb: x-axis comparison (orientation-aware)
    right_hand = lms[17].x < lms[5].x   # pinky MCP left of index MCP → right hand (mirrored)
    thumb_up   = lms[4].x < lms[3].x if right_hand else lms[4].x > lms[3].x
    result     = [thumb_up]
    for t, p in zip(tips[1:], pips[1:]):
        result.append(lms[t].y < lms[p].y)
    return result

def to_screen(x, y, fw, fh):
    sx = np.interp(x, [PADDING, fw-PADDING], [0, screen_w])
    sy = np.interp(y, [PADDING, fh-PADDING], [0, screen_h])
    return int(np.clip(sx, 0, screen_w)), int(np.clip(sy, 0, screen_h))

def draw_hand(frame, lms, fw, fh):
    pts = [px(lms, i, fw, fh) for i in range(21)]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (70, 190, 110), 1, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        col = (0, 255, 160) if i in (4, 8, 12) else (210, 210, 210)
        cv2.circle(frame, pt, 4, col, -1)
        cv2.circle(frame, pt, 5, (0,0,0), 1)

def draw_pinch_meter(frame, tip_a, tip_b, threshold, active):
    """Draw a small arc meter showing how close to pinch threshold we are."""
    d     = dist(tip_a, tip_b)
    ratio = max(0.0, min(1.0, 1.0 - (d / threshold)))      # 1 = pinched
    cx    = (tip_a[0]+tip_b[0])//2
    cy    = (tip_a[1]+tip_b[1])//2 - 22
    color = (0,255,80) if active else (255,200,60)
    cv2.ellipse(frame, (cx, cy), (14,14), -90, 0, int(360*ratio), color, 2)

def draw_guide(frame, fh):
    guides = [
        ("Index only",          "Move cursor"),
        ("Pinch (tap)",          "Left Click"),
        ("Pinch x2 quick",       "Double Click"),
        ("Pinch (hold 0.6s)",    "Drag"),
        ("V-sign (hold 0.7s)",   "Right Click"),
        ("3 fingers + move",     "Scroll"),
        ("Open palm 1s",         "Toggle ON/OFF"),
    ]
    n = len(guides)
    sy = fh - n*17 - 10
    ov = frame.copy()
    cv2.rectangle(ov, (0, sy-6), (268, fh), (0,0,0), -1)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    for i, (g, a) in enumerate(guides):
        cv2.putText(frame, f" {g}: {a}", (5, sy + i*17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190,190,190), 1, cv2.LINE_AA)

# ─── State ────────────────────────────────────────────────────────────────────
state            = "IDLE"
active           = True

# cursor smoothing
prev_x, prev_y   = 0, 0

# left-click / drag / double-click
pinch_t0         = None      # when pinch started
drag_active      = False
last_lclick      = 0.0
last_lclick2     = 0.0       # timestamp of the click before last (for double-click)
dbl_flash_t      = 0.0       # when to stop showing double-click feedback

# right-click (V-hold)
v_t0             = None      # when V-gesture started
last_rclick      = 0.0

# scroll (delta)
last_scroll_y    = None      # previous wrist pixel Y
scroll_acc       = 0.0       # float accumulator so small moves aren't wasted

# toggle
palm_t0 = None

# quit flag (set by keyboard thread)
quit_flag = threading.Event()

def _kb_listener():
    """Background thread: press Q or Esc in the console to quit."""
    print("  [Console] Press Q + Enter to quit")
    while not quit_flag.is_set():
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch.lower() in ('q', '\x1b'):
                quit_flag.set()
                break
        time.sleep(0.05)

threading.Thread(target=_kb_listener, daemon=True).start()

# ─── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    sys.exit(1)

print("=" * 56)
print("  Gesture Mouse v2  — ready")
print("=" * 56)
print("  Q / Esc  -> quit window")
print("  Emergency stop -> move mouse to any screen corner")
print("=" * 56)

fps_timer    = time.time()
frame_count  = 0
ts           = 0            # monotonic timestamp for MediaPipe

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Webcam read failed.")
        break

    frame    = cv2.flip(frame, 1)
    fh, fw   = frame.shape[:2]

    # send to MediaPipe
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img   = Image(image_format=ImageFormat.SRGB, data=rgb)
    ts      += 1
    landmarker.detect_async(mp_img, ts)

    # FPS counter
    frame_count += 1
    elapsed      = time.time() - fps_timer
    fps          = frame_count / elapsed if elapsed else 0

    # ── Header bar ────────────────────────────────────────────────────────────
    hdr = frame.copy()
    cv2.rectangle(hdr, (0,0), (fw,30), (10,10,10), -1)
    cv2.addWeighted(hdr, 0.6, frame, 0.4, 0, frame)
    hdr_col = (0,210,80) if active else (0,70,210)
    cv2.putText(frame,
                f"FPS:{fps:.0f}  {'● ACTIVE' if active else '○ PAUSED'}  [{state}]",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, hdr_col, 1, cv2.LINE_AA)

    result = _latest_result
    now    = time.time()

    if result and result.hand_landmarks:
        lms = result.hand_landmarks[0]
        draw_hand(frame, lms, fw, fh)

        f   = fingers_up(lms)
        # key points (pixel coords)
        idx_tip  = px(lms, 8,  fw, fh)
        thm_tip  = px(lms, 4,  fw, fh)
        mid_tip  = px(lms, 12, fw, fh)
        rng_tip  = px(lms, 16, fw, fh)
        wrist    = px(lms, 0,  fw, fh)

        # ── Boolean gesture flags (ordered by priority) ───────────────────────
        all_up    = all(f)                                           # open palm
        three_up  = f[1] and f[2] and f[3] and not all_up          # 3-finger scroll
        # NOTE: For mirrored webcam, f[0] (thumb) is INVERTED:
        #   f[0]=True  -> thumb is CLOSED/FOLDED
        #   f[0]=False -> thumb is OPEN/EXTENDED
        # So all thumb conditions use f[0] (not "not f[0]") to mean "thumb closed".
        v_sign    = f[1] and f[2] and f[0] and not f[3] and not f[4]  # V + thumb closed
        pinching  = dist(thm_tip, idx_tip) < PINCH_THRESHOLD          # distance pinch
        # Cursor moves ONLY when: index up + thumb CLOSED (f[0]=True) + middle down
        moving    = f[1] and f[0] and not f[2] and not pinching

        # ── TOGGLE (open palm) ────────────────────────────────────────────────
        if all_up:
            if palm_t0 is None:
                palm_t0 = now
            hold = now - palm_t0
            prog = min(hold / TOGGLE_HOLD, 1.0)
            cx, cy = fw//2, fh//2 - 40
            cv2.ellipse(frame, (cx,cy), (30,30), -90, 0, int(360*prog),
                        (0,255,180), 3)
            cv2.putText(frame, "Hold to toggle…", (cx-58, cy+52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,180), 2)
            if hold >= TOGGLE_HOLD:
                active    = not active
                palm_t0   = None
                v_t0      = None
                pinch_t0  = None
                time.sleep(0.5)
        else:
            palm_t0 = None

        # ── Everything below only runs when ACTIVE ────────────────────────────
        if active and not all_up:

            # ── SCROLL (3 fingers up, delta-based) ───────────────────────────
            if three_up:
                state = "SCROLL"
                if last_scroll_y is None:
                    last_scroll_y = wrist[1]
                    scroll_acc    = 0.0
                else:
                    dy = last_scroll_y - wrist[1]   # +ve = hand up = scroll up
                    if abs(dy) > SCROLL_DEAD_ZONE:
                        scroll_acc   += dy * SCROLL_SENS
                        amt           = int(scroll_acc)
                        if amt != 0:
                            pyautogui.scroll(amt)
                            scroll_acc -= amt   # keep fractional remainder
                    last_scroll_y = wrist[1]

                # visual indicator: arrow shows direction
                cv2.arrowedLine(frame,
                                (idx_tip[0], idx_tip[1]+30),
                                (idx_tip[0], idx_tip[1]-30),
                                (255,200,0), 2, tipLength=0.35)
                cv2.putText(frame, "SCROLL ↑↓",
                            (idx_tip[0]+12, idx_tip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
            else:
                last_scroll_y = None   # reset when not scrolling

            # ── RIGHT CLICK (V-sign hold) ─────────────────────────────────────
            if v_sign and not three_up:
                if v_t0 is None:
                    v_t0 = now
                hold  = now - v_t0
                prog  = min(hold / RIGHT_CLICK_HOLD, 1.0)
                # arc meter on middle finger tip
                cv2.ellipse(frame, mid_tip, (16,16), -90, 0, int(360*prog),
                            (140,100,255), 2)
                cv2.putText(frame, f"R-Click {int(prog*100)}%",
                            (mid_tip[0]+18, mid_tip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140,100,255), 2)
                if hold >= RIGHT_CLICK_HOLD and now - last_rclick > RIGHT_DEBOUNCE:
                    pyautogui.rightClick()
                    last_rclick = now
                    v_t0        = None
                    # flash feedback
                    cv2.circle(frame, mid_tip, 24, (140,100,255), 3)
            else:
                v_t0 = None

            # ── LEFT CLICK / DRAG (pinch) ─────────────────────────────────────
            if pinching and not three_up:
                draw_pinch_meter(frame, thm_tip, idx_tip, PINCH_THRESHOLD, True)

                if pinch_t0 is None:
                    pinch_t0 = now
                held = now - pinch_t0

                if held >= DRAG_HOLD and not drag_active:
                    drag_active = True
                    pyautogui.mouseDown()

                if drag_active:
                    state = "DRAG"
                    sx, sy = to_screen(idx_tip[0], idx_tip[1], fw, fh)
                    sx = int(prev_x + (sx - prev_x) / SMOOTHING)
                    sy = int(prev_y + (sy - prev_y) / SMOOTHING)
                    pyautogui.moveTo(sx, sy)
                    prev_x, prev_y = sx, sy
                    cv2.putText(frame, "DRAG 🖱",
                                (idx_tip[0]+10, idx_tip[1]-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,140,255), 2)
                else:
                    # Show a "charging" bar so user knows to release for click
                    bar_pct = min(held / DRAG_HOLD, 1.0)
                    bx, by  = idx_tip[0]-30, idx_tip[1]-40
                    cv2.rectangle(frame, (bx,by), (bx+60,by+8), (50,50,50), -1)
                    cv2.rectangle(frame, (bx,by), (bx+int(60*bar_pct),by+8),
                                  (0,200,80), -1)
                    cv2.putText(frame,
                                "Release=Click  Hold=Drag",
                                (bx-20, by-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

            else:
                # Pinch released
                if drag_active:
                    pyautogui.mouseUp()
                    drag_active = False
                elif pinch_t0 is not None:
                    held = now - pinch_t0
                    if held < DRAG_HOLD:
                        time_since_last = now - last_lclick
                        # Double-click: second tap arrives within window
                        # Only needs a minimal 0.06s gap (not full debounce)
                        if 0.06 < time_since_last < DBLCLICK_WINDOW and last_lclick > 0:
                            pyautogui.doubleClick()
                            dbl_flash_t  = now + 0.5
                            last_lclick  = 0.0   # reset to prevent triple-click
                        # Single click: first tap, needs full debounce
                        elif time_since_last > CLICK_DEBOUNCE:
                            pyautogui.click()
                            last_lclick  = now
                    draw_pinch_meter(frame, thm_tip, idx_tip, PINCH_THRESHOLD, False)
                pinch_t0 = None

            # ── Double-click flash feedback ───────────────────────────────────
            if now < dbl_flash_t:
                cv2.circle(frame, idx_tip, 30, (0, 220, 255), 3)
                cv2.circle(frame, idx_tip, 18, (0, 220, 255), 2)
                cv2.putText(frame, "DOUBLE CLICK!",
                            (idx_tip[0]+10, idx_tip[1]-16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            elif pinch_t0 is None and not drag_active and now - last_lclick < 0.25:
                # single click confirmation flash
                cv2.circle(frame, idx_tip, 24, (0,255,80), 3)
                cv2.putText(frame, "CLICK",
                            (idx_tip[0]+10, idx_tip[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,80), 2)

            # ── MOVE (index only) ─────────────────────────────────────────────
            if moving and not three_up and not pinching and not drag_active:
                state = "MOVE"
                sx, sy = to_screen(idx_tip[0], idx_tip[1], fw, fh)
                sx = int(prev_x + (sx - prev_x) / SMOOTHING)
                sy = int(prev_y + (sy - prev_y) / SMOOTHING)
                pyautogui.moveTo(sx, sy)
                prev_x, prev_y = sx, sy
                cv2.circle(frame, idx_tip, 9,  (0,220,255), -1)
                cv2.circle(frame, idx_tip, 11, (255,255,255), 1)

            if not moving and not three_up and not pinching and not drag_active and not v_sign:
                state = "IDLE"

            # ── Live thumb–index distance display ─────────────────────────────
            d_now = dist(thm_tip, idx_tip)
            cv2.line(frame, thm_tip, idx_tip,
                     (0,255,80) if pinching else (255,200,60), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{d_now:.0f}/{PINCH_THRESHOLD}px",
                        (min(thm_tip[0],idx_tip[0]), min(thm_tip[1],idx_tip[1])-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180,180,180), 1)

    else:
        # no hand
        state = "IDLE"
        palm_t0 = v_t0 = None
        last_scroll_y = None
        if drag_active:
            pyautogui.mouseUp()
            drag_active = False
        cv2.putText(frame, "No hand detected",
                    (fw//2-90, fh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,50,190), 2, cv2.LINE_AA)

    draw_guide(frame, fh)
    cv2.imshow("Gesture Mouse v2", frame)
    if quit_flag.is_set() or (cv2.waitKey(1) & 0xFF in (ord("q"), 27)):
        break

# ─── Cleanup ──────────────────────────────────────────────────────────────────
if drag_active:
    pyautogui.mouseUp()
landmarker.close()
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("[INFO] Gesture Mouse stopped.")
sys.exit(0)
