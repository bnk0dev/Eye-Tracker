import cv2
import mediapipe as mp
import time
import tkinter as tk
import threading
import ctypes
import numpy as np
import math

# --- Makine Öğrenmesi Kütüphaneleri ---
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
        
        # Ekranı şeffaf ve tıklanamaz yap
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.config(bg='white')
        self.root.attributes('-transparentcolor', 'white')
        
        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h, bg='white', highlightthickness=0)
        self.canvas.pack()
        
        # İMLEÇ BÜYÜTÜLDÜ (15 -> 25)
        self.r = 25 
        self.bubble = self.canvas.create_oval(
            self.screen_w/2 - self.r, self.screen_h/2 - self.r, 
            self.screen_w/2 + self.r, self.screen_h/2 + self.r, 
            fill='red', outline='yellow', width=3
        )
        
        self.calibration_text = self.canvas.create_text(
            self.screen_w/2, 100, text="", fill="red", font=("Arial", 28, "bold"), justify="center"
        )
        
        self.target_x = self.screen_w / 2
        self.target_y = self.screen_h / 2
        self.smooth_x = self.screen_w / 2
        self.smooth_y = self.screen_h / 2
        self.calibration_mode = False
        
        self.root.after(100, self.make_clickthrough)
        self.update_bubble()
        
    def make_clickthrough(self):
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            set_clickthrough(hwnd)
        except Exception:
            pass
            
    def update_gaze(self, x, y):
        if not self.calibration_mode:
            self.target_x = x
            self.target_y = y
            
    def set_calibration_target(self, x, y, message):
        self.calibration_mode = True
        self.target_x = x
        self.target_y = y
        self.canvas.itemconfig(self.calibration_text, text=message)
        
    def end_calibration(self):
        self.calibration_mode = False
        self.canvas.itemconfig(self.calibration_text, text="")
        
    def update_bubble(self):
        # Dinamik Yumuşatma: Mesafe uzaksa hızlı git, yakınsa titremeyi durdurmak için yavaşla
        dist = math.hypot(self.target_x - self.smooth_x, self.target_y - self.smooth_y)
        dynamic_smooth = 0.2 if dist > 50 else 0.05
        
        self.smooth_x += (self.target_x - self.smooth_x) * dynamic_smooth
        self.smooth_y += (self.target_y - self.smooth_y) * dynamic_smooth
        
        self.canvas.coords(self.bubble, 
                           self.smooth_x - self.r, self.smooth_y - self.r, 
                           self.smooth_x + self.r, self.smooth_y + self.r)
                           
        self.root.after(16, self.update_bubble) # ~60 FPS

# Göz içindeki lokal pozisyonu hesaplayan fonksiyon
def get_eye_ratio(iris_pts, inner_pt, outer_pt, top_pt, bottom_pt):
    iris_x = sum([pt.x for pt in iris_pts]) / len(iris_pts)
    iris_y = sum([pt.y for pt in iris_pts]) / len(iris_pts)
    
    eye_width = math.hypot(outer_pt.x - inner_pt.x, outer_pt.y - inner_pt.y)
    eye_height = math.hypot(top_pt.x - bottom_pt.x, top_pt.y - bottom_pt.y)
    
    if eye_width == 0 or eye_height == 0: return 0.5, 0.5
        
    dx = abs(iris_x - inner_pt.x) / eye_width
    dy = abs(iris_y - top_pt.y) / eye_height
    return dx, dy

def cv2_worker(overlay):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cam = cv2.VideoCapture(0)
    screen_w = overlay.screen_w
    screen_h = overlay.screen_h
    
    calibrating = False
    calibration_step = 0
    calibration_features_src = [] # Artık 4 boyutlu veri tutacak (Göz X, Göz Y, Baş Yaw, Baş Pitch)
    
    is_recording = False
    frames_to_record = 30
    recorded_frames_data = []

    offset = 50
    cw = screen_w // 2
    ch = screen_h // 2
    calibration_targets = [
        (offset, offset),               
        (cw, offset),                   
        (screen_w - offset, offset),    
        (offset, ch),                   
        (cw, ch),                       
        (screen_w - offset, ch),        
        (offset, screen_h - offset),    
        (cw, screen_h - offset),        
        (screen_w - offset, screen_h - offset) 
    ]
    
    svr_model_x = None
    svr_model_y = None
    
    # EMA Smoothing değişkenleri
    rel_x_history, rel_y_history = 0.0, 0.0
    yaw_history, pitch_history = 0.0, 0.0
    first_frame = True
    alpha = 0.15 

    left_iris_idx = [474, 475, 476, 477]
    right_iris_idx = [469, 470, 471, 472]
    l_inner = 362; l_outer = 263; l_top = 386; l_bottom = 374
    r_inner = 133; r_outer = 33; r_top = 159; r_bottom = 145

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. GÖZ ORANLARI (İrisin göz içindeki konumu)
            l_dx, l_dy = get_eye_ratio(
                [landmarks[i] for i in left_iris_idx],
                landmarks[l_inner], landmarks[l_outer],
                landmarks[l_top], landmarks[l_bottom]
            )
            r_dx, r_dy = get_eye_ratio(
                [landmarks[i] for i in right_iris_idx],
                landmarks[r_inner], landmarks[r_outer],
                landmarks[r_top], landmarks[r_bottom]
            )
            raw_rel_x = (l_dx + r_dx) / 2
            raw_rel_y = (l_dy + r_dy) / 2
            
            # 2. BAŞ DURUŞU AÇISI (Head Pose - Piyasada kullanılan sır)
            nose_tip = landmarks[1]
            face_center = landmarks[168] # İki kaşın arası
            
            # Burun ucu ile yüz merkezinin X ve Y farkı, başın dönüş açısını (Yaw ve Pitch) verir
            raw_head_yaw = nose_tip.x - face_center.x
            raw_head_pitch = nose_tip.y - face_center.y
            
            # Pürüzsüzleştirme (Tüm özellikler için)
            if first_frame:
                rel_x_history, rel_y_history = raw_rel_x, raw_rel_y
                yaw_history, pitch_history = raw_head_yaw, raw_head_pitch
                first_frame = False
            else:
                rel_x_history = (alpha * raw_rel_x) + ((1.0 - alpha) * rel_x_history)
                rel_y_history = (alpha * raw_rel_y) + ((1.0 - alpha) * rel_y_history)
                yaw_history = (alpha * raw_head_yaw) + ((1.0 - alpha) * yaw_history)
                pitch_history = (alpha * raw_head_pitch) + ((1.0 - alpha) * pitch_history)
            
            # O anki 4 Boyutlu Yapay Zeka Özelliklerimiz (Features)
            current_features = [rel_x_history, rel_y_history, yaw_history, pitch_history]

            # Göz piksellerini yeşille çiz
            for idx in left_iris_idx + right_iris_idx:
                pt = landmarks[idx]
                cv2.circle(frame, (int(pt.x * frame_w), int(pt.y * frame_h)), 2, (0, 255, 0), -1)

            # EĞER KAYIT MODUNDAYSA (Boşluk tuşuna basıldıysa)
            if is_recording:
                recorded_frames_data.append(current_features)
                frames_to_record -= 1
                
                cv2.putText(frame, f"KAYDEDILIYOR... {frames_to_record}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                
                if frames_to_record <= 0:
                    # 4 özelliğin de 30 karelik Medyan'ını alıyoruz (Kusursuz ölçüm)
                    med_features = np.median(recorded_frames_data, axis=0)
                    
                    calibration_features_src.append(med_features)
                    print(f"Adım {calibration_step+1} Kaydedildi (Göz + Baş Açısı)")
                    
                    calibration_step += 1
                    is_recording = False
                    
                    if calibration_step >= 9:
                        calibrating = False
                        overlay.end_calibration()
                        
                        # --- BAŞ AÇISI DESTEKLİ SVR EĞİTİMİ ---
                        print("4 Boyutlu SVR Modeli Eğitiliyor (Baş Telafisi Devrede)...")
                        X_train = np.array(calibration_features_src) # [Göz_X, Göz_Y, Baş_Yaw, Baş_Pitch]
                        y_train_x = np.array([tgt[0] for tgt in calibration_targets])
                        y_train_y = np.array([tgt[1] for tgt in calibration_targets])
                        
                        try:
                            # C ve Epsilon değerleri baş açısını da işleyebilmesi için optimize edildi
                            svr_model_x = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=500, gamma='scale', epsilon=10.0))
                            svr_model_y = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=500, gamma='scale', epsilon=10.0))
                            
                            svr_model_x.fit(X_train, y_train_x)
                            svr_model_y.fit(X_train, y_train_y)
                            print("Gelişmiş Kalibrasyon Başarılı!")
                        except Exception as e:
                            print(f"SVR Kalibrasyon hatası: {e}")
                            svr_model_x, svr_model_y = None, None
                    else:
                        target = calibration_targets[calibration_step]
                        overlay.set_calibration_target(target[0], target[1], f"Topa Bakin!\nAdim {calibration_step+1}/9 (SPACE basin)")
            
            # --- TAHMİN BÖLÜMÜ ---
            elif not calibrating and svr_model_x is not None and svr_model_y is not None:
                # Makine öğrenmesi modeline 4 veriyi de verip ekranın neresine baktığını tahmin ediyoruz
                target_x = svr_model_x.predict([current_features])[0]
                target_y = svr_model_y.predict([current_features])[0]
                
                # Ekran sınırlarına kelepçele
                target_x = max(0, min(screen_w, target_x))
                target_y = max(0, min(screen_h, target_y))
                overlay.update_gaze(target_x, target_y)
            
            elif not calibrating:
                cv2.putText(frame, "Kalibrasyon Bekleniyor (C'ye basin)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Adim {calibration_step+1}/9: Topa bakip SPACE basin", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Eye Tracker', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('c') and not calibrating and not is_recording:
            calibrating = True
            calibration_step = 0
            calibration_features_src = []
            target = calibration_targets[0]
            overlay.set_calibration_target(target[0], target[1], f"Topa Bakin!\nAdim 1/9 (SPACE basin)")
            print("-- Gelişmiş Kalibrasyon Basladi --")
        elif key == ord(' ') and calibrating and not is_recording:
            is_recording = True
            frames_to_record = 30
            recorded_frames_data = []

    cam.release()
    cv2.destroyAllWindows()
    overlay.root.after(0, overlay.root.destroy)

if __name__ == "__main__":
    overlay = GazeOverlay()
    cv_thread = threading.Thread(target=cv2_worker, args=(overlay,), daemon=True)
    cv_thread.start()
    overlay.root.mainloop()