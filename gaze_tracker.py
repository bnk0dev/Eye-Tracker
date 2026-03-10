import cv2
import mediapipe as mp
import time
import tkinter as tk
import threading
import ctypes
import numpy as np

# Windows API Constants for click-through window
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
GWL_EXSTYLE = -20

def set_clickthrough(hwnd):
    try:
        user32 = ctypes.windll.user32
        styles = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, styles | WS_EX_LAYERED | WS_EX_TRANSPARENT)
    except Exception as e:
        print(f"Could not set click-through: {e}")

class GazeOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gaze Bubble")
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        
        # Make window transparent, topmost, and borderless
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.config(bg='white')
        
        # 'white' pixels will be fully transparent
        self.root.attributes('-transparentcolor', 'white')
        
        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h, bg='white', highlightthickness=0)
        self.canvas.pack()
        
        self.r = 15 # Bubble radius
        self.bubble = self.canvas.create_oval(
            self.screen_w/2 - self.r, self.screen_h/2 - self.r, 
            self.screen_w/2 + self.r, self.screen_h/2 + self.r, 
            fill='red', outline='yellow', width=2
        )
        
        self.calibration_text = self.canvas.create_text(
            self.screen_w/2, 100, text="", fill="red", font=("Arial", 28, "bold"), justify="center"
        )
        
        self.target_x = self.screen_w / 2
        self.target_y = self.screen_h / 2
        self.smooth_x = self.screen_w / 2
        self.smooth_y = self.screen_h / 2
        self.smooth_factor = 0.05  # Reduced for more stability (from 0.15)
        self.calibration_mode = False
        
        # After window creation, make it click-through
        self.root.after(100, self.make_clickthrough)
        
        self.update_bubble()
        
    def make_clickthrough(self):
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            set_clickthrough(hwnd)
        except Exception:
            pass
        
    def update_gaze(self, x, y):
        # Do not allow cv2 thread to steal coordinate focus while calibrating
        if not self.calibration_mode:
            self.target_x = x
            self.target_y = y
            self.canvas.itemconfig(self.calibration_text, text="")
            
    def set_calibration_target(self, x, y, message):
        self.calibration_mode = True
        self.target_x = x
        self.target_y = y
        self.canvas.itemconfig(self.calibration_text, text=message)
        
    def end_calibration(self):
        self.calibration_mode = False
        self.canvas.itemconfig(self.calibration_text, text="")
        
    def update_bubble(self):
        # Smooth interpolation
        self.smooth_x += (self.target_x - self.smooth_x) * self.smooth_factor
        self.smooth_y += (self.target_y - self.smooth_y) * self.smooth_factor
        
        # Move bubble
        self.canvas.coords(self.bubble, 
                           self.smooth_x - self.r, self.smooth_y - self.r, 
                           self.smooth_x + self.r, self.smooth_y + self.r)
                           
        # Repeat every ~16ms for 60FPS
        self.root.after(16, self.update_bubble)

# Fallback values for before calibration (arbitrarily mapped for face-width normalization)
HORIZ_MIN = -0.06
HORIZ_MAX = 0.06
VERT_MIN = -0.02
VERT_MAX = 0.04

def cv2_worker(overlay):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cam = cv2.VideoCapture(0)
    
    print("Starting Gaze Tracker.")
    print("--- INSTRUCTIONS ---")
    print("1. Look at the camera window, press 'c' to Start Calibration.")
    print("2. A red bubble will move to 4 corners of the screen.")
    print("3. Look directly at the bubble and press 'SPACE' on the OpenCV window to lock the step.")
    print("Press 'q' or 'ESC' on the OpenCV window to quit.")

    screen_w = overlay.screen_w
    screen_h = overlay.screen_h
    
    calibrating = False
    calibration_step = 0
    calibration_points_src = []
    
    # Target points slightly inset from the true corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    offset = 50
    calibration_targets = [
        (offset, offset),
        (screen_w - offset, offset),
        (screen_w - offset, screen_h - offset),
        (offset, screen_h - offset)
    ]
    
    transform_matrix = None

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break
            
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        rel_x = 0
        rel_y = 0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Left/Right eye indices mapping
            left_iris_idx = [474, 475, 476, 477]
            right_iris_idx = [469, 470, 471, 472]
            
            # Overall face bounding landmarks
            face_left_pt = landmarks[234]  # Leftmost face point
            face_right_pt = landmarks[454] # Rightmost face point
            face_top_pt = landmarks[10]    # Top forehead
            face_bottom_pt = landmarks[152] # Bottom chin
            
            face_w = abs(face_right_pt.x - face_left_pt.x)
            face_h = abs(face_bottom_pt.y - face_top_pt.y)
            
            # Only proceed if we clearly see the face
            if face_w > 0 and face_h > 0:
                left_iris_pts = [landmarks[i] for i in left_iris_idx]
                right_iris_pts = [landmarks[i] for i in right_iris_idx]
                
                left_iris_x = sum([pt.x for pt in left_iris_pts]) / len(left_iris_pts)
                left_iris_y = sum([pt.y for pt in left_iris_pts]) / len(left_iris_pts)
                
                right_iris_x = sum([pt.x for pt in right_iris_pts]) / len(right_iris_pts)
                right_iris_y = sum([pt.y for pt in right_iris_pts]) / len(right_iris_pts)
                
                # Inner eye corner landmarks
                left_inner = landmarks[362]
                right_inner = landmarks[133]
                
                # New normalization: distance from inner eye corner scaled by entire face width/height
                # This naturally compensates for leaning in/out towards camera.
                left_dx = (left_iris_x - left_inner.x) / face_w
                left_dy = (left_iris_y - left_inner.y) / face_h
                
                right_dx = (right_iris_x - right_inner.x) / face_w
                right_dy = (right_iris_y - right_inner.y) / face_h
                
                rel_x = (left_dx + right_dx) / 2
                rel_y = (left_dy + right_dy) / 2
            
                # Draw on frame
                for pt in left_iris_pts + right_iris_pts:
                    x = int(pt.x * frame_w)
                    y = int(pt.y * frame_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                cv2.putText(frame, f"rel_x: {rel_x:.4f} rel_y: {rel_y:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if not calibrating:
                    if transform_matrix is not None:
                        # Map with non-linear perspective transform
                        src_pts = np.array([[[rel_x, rel_y]]], dtype=np.float32)
                        dst_pts = cv2.perspectiveTransform(src_pts, transform_matrix)
                        target_x, target_y = dst_pts[0][0]
                        
                        target_x = max(0, min(screen_w, target_x))
                        target_y = max(0, min(screen_h, target_y))
                        overlay.update_gaze(target_x, target_y)
                        
                        cv2.putText(frame, f"Screen X: {int(target_x)} Y: {int(target_y)} (Mapped)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Fallback mapping
                        norm_x = (rel_x - HORIZ_MIN) / (HORIZ_MAX - HORIZ_MIN) if HORIZ_MAX != HORIZ_MIN else 0.5
                        norm_y = (rel_y - VERT_MIN) / (VERT_MAX - VERT_MIN) if VERT_MAX != VERT_MIN else 0.5
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))
                        
                        target_x = screen_w * norm_x
                        target_y = screen_h * norm_y
                        overlay.update_gaze(target_x, target_y)
                        
                        cv2.putText(frame, "Waiting for Calibration (Press C)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"Calibrating Step {calibration_step+1}/4", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, "Look at circle & press SPACE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Gaze Tracker Camera', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('c') and not calibrating:
            calibrating = True
            calibration_step = 0
            calibration_points_src = []
            target = calibration_targets[calibration_step]
            overlay.set_calibration_target(target[0], target[1], f"Look Here!\nStep 1/4 (Press Space in Camera Window)")
            print(f"-- Calibration Started --")
        elif key == ord(' ') and calibrating:
            # Record Point
            print(f"Recorded Step {calibration_step+1}: rel_x={rel_x:.4f}, rel_y={rel_y:.4f}")
            calibration_points_src.append([rel_x, rel_y])
            calibration_step += 1
            
            if calibration_step >= 4:
                # Calculate matrix
                calibrating = False
                overlay.end_calibration()
                
                src_pts = np.array(calibration_points_src, dtype=np.float32)
                dst_pts = np.array(calibration_targets, dtype=np.float32)
                
                try:
                    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    print("Calibration Successful! Resulting Matrix:")
                    print(transform_matrix)
                except Exception as e:
                    print(f"Calibration failed: {e}")
                    transform_matrix = None
            else:
                target = calibration_targets[calibration_step]
                overlay.set_calibration_target(target[0], target[1], f"Look Here!\nStep {calibration_step+1}/4 (Press Space)")

    cam.release()
    cv2.destroyAllWindows()
    # Safely instruct tkinter to stop
    overlay.root.after(0, overlay.root.destroy)

if __name__ == "__main__":
    overlay = GazeOverlay()
    
    # Run OpenCV gathering natively in a background thread
    cv_thread = threading.Thread(target=cv2_worker, args=(overlay,), daemon=True)
    cv_thread.start()
    
    # Start Tkinter mainloop on the main thread (needs to be main thread for Windows GUIs)
    overlay.root.mainloop()

