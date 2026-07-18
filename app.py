from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime
import numpy as np
import requests
import threading
import sqlite3
import math
from collections import deque
import mediapipe as mp
import json
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# =====================================================================
# 1. TELEGRAM BOT CONFIG
# =====================================================================

TELEGRAM_BOTS = [
    {
        "token": "8743971848:AAGNg27W1UTrcAnk4X02JjMIpK6AYKkvXPw",
        "chat_id": "6786276785"
    },
    {
        "token": "8587935902:AAHOIkb9c9uP9ifjDtXVMaTjv3HophnO1fY",
        "chat_id": "8751919066"
    },
]

# =====================================================================
# 2. FUNGSI KIRIM TELEGRAM
# =====================================================================

def kirim_notifikasi_telegram(foto_path, pesan_teks, kamera_nama="Webcam"):
    def proses_kirim():
        try:
            if not os.path.exists(foto_path):
                print(f"❌ [TELEGRAM] File foto tidak ditemukan")
                return
            
            for idx, bot in enumerate(TELEGRAM_BOTS, 1):
                try:
                    token = bot["token"]
                    chat_id = bot["chat_id"]
                    
                    url_foto = f"https://api.telegram.org/bot{token}/sendPhoto"
                    
                    with open(foto_path, 'rb') as foto_file:
                        payload = {
                            'chat_id': chat_id,
                            'caption': pesan_teks,
                            'parse_mode': 'HTML'
                        }
                        files = {'photo': foto_file}
                        
                        response = requests.post(url_foto, data=payload, files=files, timeout=30)
                        
                        if response.status_code == 200:
                            print(f"✅ [TELEGRAM] Berhasil dikirim ke Bot {idx}")
                        else:
                            print(f"❌ [TELEGRAM] Gagal kirim ke Bot {idx}")
                            
                except Exception as e:
                    print(f"❌ [TELEGRAM] Error Bot {idx}: {e}")
                    
        except Exception as e:
            print(f"❌ [TELEGRAM] Error utama: {e}")

    thread = threading.Thread(target=proses_kirim, daemon=True)
    thread.start()
    return thread

# =====================================================================
# 3. SETUP FOLDER & DATABASE
# =====================================================================

output_folder = os.path.join(os.path.dirname(__file__), 'violators')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

db_path = os.path.join(os.path.dirname(__file__), 'log_pelanggaran.db')
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        waktu TEXT,
        jenis_pelanggaran TEXT,
        kamera_sumber TEXT,
        lokasi_foto TEXT,
        detail_deteksi TEXT
    )
''')
conn.commit()
print("✅ Database SQLite siap")

# =====================================================================
# 4. LOAD MODEL YOLO
# =====================================================================

try:
    model_objek = YOLO('models/cigaret_vape.pt')
    model_asap = YOLO('models/smoke_only.pt')
    model_human = YOLO('yolov8n.pt')
    print("✅ Semua model berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    exit()

# =====================================================================
# 5. MEDIAPIPE FACE MESH
# =====================================================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================================
# 6. KONFIGURASI
# =====================================================================

YOLO_CONF_OBJEK = 0.35
YOLO_CONF_ASAP = 0.30
YOLO_CONF_HUMAN = 0.30
YOLO_IMGSZ = 640
MOUTH_DISTANCE_THRESHOLD = 150
DETECTION_BUFFER = 5
jeda_notifikasi = 10

# State untuk kamera
camera_state = {
    'mode': 'single',  # 'single' atau 'dual'
    'selected_camera': 'internal',  # 'internal' atau 'external' (untuk mode single)
    'internal_active': False,
    'external_active': False,
    'running': False
}

# Buffer per kamera
buffers = {
    'internal': {
        'human': deque(maxlen=DETECTION_BUFFER),
        'smoke': deque(maxlen=DETECTION_BUFFER),
        'objek': deque(maxlen=DETECTION_BUFFER)
    },
    'external': {
        'human': deque(maxlen=DETECTION_BUFFER),
        'smoke': deque(maxlen=DETECTION_BUFFER),
        'objek': deque(maxlen=DETECTION_BUFFER)
    }
}

# Timer notifikasi
last_notification = {
    'internal': 0,
    'external': 0
}

# Frame terakhir untuk ditampilkan
latest_frames = {
    'internal': None,
    'external': None
}

# Statistik realtime (sinkron dengan overlay video)
current_stats = {
    'internal': {
        'human': 0,
        'objek': 0,
        'smoke': 0
    },
    'external': {
        'human': 0,
        'objek': 0,
        'smoke': 0
    }
}

# Logs
logs_data = {
    'internal': [],
    'external': []
}
MAX_LOGS = 100

# =====================================================================
# 7. FUNGSI DETEKSI
# =====================================================================

def get_mouth_position(roi):
    try:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = roi.shape[:2]
            
            mouth_left = landmarks.landmark[61]
            mouth_right = landmarks.landmark[291]
            mouth_top = landmarks.landmark[13]
            mouth_bottom = landmarks.landmark[14]
            
            cx = int((mouth_left.x + mouth_right.x) / 2 * w)
            cy = int((mouth_top.y + mouth_bottom.y) / 2 * h)
            
            return (cx, cy)
    except:
        pass
    return None

def is_smoking_near_mouth(smoking_center, person_bbox, frame):
    px1, py1, px2, py2 = person_bbox
    
    head_y2 = py1 + int((py2 - py1) * 0.4)
    roi_head = frame[py1:head_y2, px1:px2]
    
    if roi_head.size == 0:
        return False, 999
    
    mouth_pos = get_mouth_position(roi_head)
    
    if mouth_pos is None:
        return False, 999
    
    mouth_global = (px1 + mouth_pos[0], py1 + mouth_pos[1])
    
    sx, sy = smoking_center
    distance = math.sqrt((sx - mouth_global[0])**2 + (sy - mouth_global[1])**2)
    
    # Gambar titik mulut (tanpa teks)
    cv2.circle(frame, mouth_global, 8, (0, 255, 0), -1)
    cv2.circle(frame, mouth_global, 10, (0, 255, 0), 1)
    
    if distance < MOUTH_DISTANCE_THRESHOLD:
        cv2.line(frame, mouth_global, smoking_center, (0, 0, 255), 2)
    
    return distance < MOUTH_DISTANCE_THRESHOLD, distance

def detect_all_objects_with_mouth(frame):
    results = {
        'human': [],
        'rokok': [],
        'asap': []
    }
    
    # Deteksi Manusia
    try:
        human_results = model_human(frame, conf=YOLO_CONF_HUMAN, imgsz=YOLO_IMGSZ, verbose=False)
        for r in human_results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        conf = float(box.conf[0])
                        if conf >= YOLO_CONF_HUMAN:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            results['human'].append({
                                'bbox': [x1, y1, x2, y2],
                                'conf': conf,
                                'center': ((x1+x2)//2, (y1+y2)//2),
                                'id': len(results['human']) + 1
                            })
    except Exception as e:
        pass
    
    # Deteksi Rokok/Vape
    try:
        rokok_results = model_objek(frame, conf=YOLO_CONF_OBJEK, imgsz=YOLO_IMGSZ, verbose=False)
        for r in rokok_results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf >= YOLO_CONF_OBJEK:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        results['rokok'].append({
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'center': ((x1+x2)//2, (y1+y2)//2),
                            'class': model_objek.names[cls]
                        })
    except Exception as e:
        pass
    
    # Deteksi Asap
    try:
        asap_results = model_asap(frame, conf=YOLO_CONF_ASAP, imgsz=YOLO_IMGSZ, verbose=False)
        for r in asap_results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf >= YOLO_CONF_ASAP:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        results['asap'].append({
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'center': ((x1+x2)//2, (y1+y2)//2)
                        })
    except Exception as e:
        pass
    
    # Cek rokok dekat mulut
    for person in results['human']:
        person_bbox = person['bbox']
        person['near_mouth'] = False
        person['near_mouth_distance'] = 999
        person['has_rokok'] = False
        person['has_asap'] = False
        
        for rokok in results['rokok']:
            is_near, dist = is_smoking_near_mouth(rokok['center'], person_bbox, frame)
            if is_near:
                person['near_mouth'] = True
                person['near_mouth_distance'] = dist
                person['has_rokok'] = True
                break
        
        for asap in results['asap']:
            ax, ay = asap['center']
            px1, py1, px2, py2 = person_bbox
            padding = 50
            if (px1 - padding < ax < px2 + padding) and (py1 - padding < ay < py2 + padding):
                person['has_asap'] = True
                break
    
    return results

def draw_detections_with_id(frame, results):
    COLORS = {
        'human': (255, 0, 0),
        'rokok': (0, 0, 255),
        'asap': (0, 255, 255)
    }
    
    for det in results['human']:
        x1, y1, x2, y2 = det['bbox']
        conf = det['conf']
        person_id = det.get('id', 0)
        near_mouth = det.get('near_mouth', False)
        
        color = (0, 0, 255) if near_mouth else COLORS['human']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        label = f"ORANG #{person_id} {conf:.2f}"
        if near_mouth:
            label += " 🔥"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if near_mouth:
            cv2.putText(frame, "ROKOK DEKAT MULUT!", (x1, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    for det in results['rokok']:
        x1, y1, x2, y2 = det['bbox']
        conf = det['conf']
        class_name = det['class']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['rokok'], 2)
        label = f"{class_name.upper()} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLORS['rokok'], -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cx, cy = det['center']
        cv2.circle(frame, (cx, cy), 5, COLORS['rokok'], -1)
    
    for det in results['asap']:
        x1, y1, x2, y2 = det['bbox']
        conf = det['conf']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['asap'], 2)
        label = f"ASAP {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLORS['asap'], -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

def check_violation_with_mouth(results):
    has_human = len(results['human']) > 0
    has_rokok = len(results['rokok']) > 0
    has_asap = len(results['asap']) > 0
    has_near_mouth = any(person.get('near_mouth', False) for person in results['human'])
    
    if has_human and not has_rokok and not has_asap:
        return 'normal', "✅ Tidak ada pelanggaran"
    elif has_human and not has_rokok and has_asap:
        return 'warning_asap', "⚠️ PERINGATAN: Terdeteksi asap"
    elif has_human and has_rokok and not has_asap:
        return 'warning_rokok', "⚠️ PERINGATAN: Terdeteksi rokok/vape"
    elif has_human and has_near_mouth and has_rokok and has_asap:
        return 'violation', "🚨 PELANGGARAN: Merokok dekat mulut!"
    elif has_human and has_rokok and has_asap and not has_near_mouth:
        return 'warning_rokok', "⚠️ PERINGATAN: Rokok tidak dekat mulut"
    else:
        return 'no_human', "Tidak ada manusia terdeteksi"

def create_telegram_message(results, status, status_text, kamera_nama="Webcam"):
    now = datetime.now()
    waktu_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    total_human = len(results['human'])
    total_rokok = len(results['rokok'])
    total_asap = len(results['asap'])
    
    message = f"🔔 [SMOKING DETECTED] 🔔\n"
    message += f"━━━━━━━━━━━━━━━━━━━\n"
    message += f"🚨 {status_text}\n"
    message += f"━━━━━━━━━━━━━━━━━━━\n"
    message += f"📸 Kamera: {kamera_nama}\n"
    message += f"📅 Waktu: {waktu_str}\n"
    message += f"━━━━━━━━━━━━━━━━━━━\n"
    message += f"📊 DETAIL DETEKSI:\n"
    message += f"   • Manusia Terdeteksi: {total_human} orang\n"
    message += f"   • Rokok/Vape: {total_rokok} objek\n"
    message += f"   • Asap: {total_asap} objek\n"
    
    if total_human > 0:
        message += f"\n👤 Detail Orang:\n"
        for person in results['human']:
            person_id = person.get('id', 0)
            near_mouth = person.get('near_mouth', False)
            near_distance = person.get('near_mouth_distance', 999)
            has_rokok = person.get('has_rokok', False)
            has_asap = person.get('has_asap', False)
            
            message += f"\n👤 Orang #{person_id}:\n"
            
            closest_rokok_conf = 0
            for rokok in results['rokok']:
                hx, hy = person['center']
                rx, ry = rokok['center']
                distance = math.sqrt((hx - rx)**2 + (hy - ry)**2)
                if distance < 300:
                    if rokok['conf'] > closest_rokok_conf:
                        closest_rokok_conf = rokok['conf']
            
            closest_asap_conf = 0
            for asap in results['asap']:
                hx, hy = person['center']
                ax, ay = asap['center']
                distance = math.sqrt((hx - ax)**2 + (hy - ay)**2)
                if distance < 300:
                    if asap['conf'] > closest_asap_conf:
                        closest_asap_conf = asap['conf']
            
            if has_rokok and near_mouth:
                message += f"   • Rokok/Vape: ✅ TERDETEKSI DEKAT MULUT!\n"
                message += f"     (Confidence: {closest_rokok_conf:.2f}, Jarak: {near_distance:.0f}px)\n"
            elif has_rokok:
                message += f"   • Rokok/Vape: ✅ TERDETEKSI!\n"
                message += f"     (Confidence: {closest_rokok_conf:.2f})\n"
            else:
                message += f"   • Rokok/Vape: ❌ Tidak terdeteksi\n"
            
            if has_asap:
                message += f"   • Asap: ✅ TERDETEKSI!\n"
                message += f"     (Confidence: {closest_asap_conf:.2f})\n"
            else:
                message += f"   • Asap: ❌ Tidak terdeteksi\n"
            
            if near_mouth:
                message += f"   • 📍 Jarak ke mulut: {near_distance:.0f}px (DEKAT!)\n"
            else:
                message += f"   • 📍 Jauh dari mulut\n"
    
    message += f"\n━━━━━━━━━━━━━━━━━━━\n"
    message += f"📎 Lampiran foto terlampir\n"
    message += f"⚙️ Confidence Threshold:\n"
    message += f"   • Rokok: {YOLO_CONF_OBJEK}\n"
    message += f"   • Asap: {YOLO_CONF_ASAP}\n"
    message += f"   • Jarak mulut: {MOUTH_DISTANCE_THRESHOLD}px"
    
    return message

def process_frame(frame, kamera_nama, buffer_key):
    if frame is None:
        return None, None
    
    results = detect_all_objects_with_mouth(frame)

    # Update statistik realtime untuk dashboard
    current_stats[buffer_key]['human'] = len(results['human'])
    current_stats[buffer_key]['objek'] = len(results['rokok'])
    current_stats[buffer_key]['smoke'] = len(results['asap'])

    frame = draw_detections_with_id(frame, results)
    
    has_human = len(results['human']) > 0
    has_rokok = len(results['rokok']) > 0
    has_asap = len(results['asap']) > 0
    
    buffers[buffer_key]['human'].append(has_human)
    buffers[buffer_key]['smoke'].append(has_asap)
    buffers[buffer_key]['objek'].append(has_rokok)
    
    status, status_text = check_violation_with_mouth(results)
    
    # Info panel
    h, w, _ = frame.shape
    info_bg = np.zeros((200, 400, 3), dtype=np.uint8)
    info_bg[:] = (0, 0, 0)
    info_bg = cv2.addWeighted(info_bg, 0.5, frame[10:210, 10:410], 0.5, 0)
    frame[10:210, 10:410] = info_bg
    
    y_pos = 30
    cv2.putText(frame, f"📹 {kamera_nama}", (20, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_status = y_pos + 35
    status_color = (0, 255, 0)
    if status == 'normal':
        status_color = (0, 255, 0)
    elif status in ['warning_asap', 'warning_rokok']:
        status_color = (0, 255, 255)
    elif status == 'violation':
        status_color = (0, 0, 255)
    
    cv2.putText(frame, status_text, (20, y_status),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    stats = [
        f"👤 Manusia: {len(results['human'])}",
        f"🚬 Rokok: {len(results['rokok'])}",
        f"💨 Asap: {len(results['asap'])}",
        f"📊 Total: {len(results['human']) + len(results['rokok']) + len(results['asap'])}"
    ]
    
    for i, stat in enumerate(stats):
        cv2.putText(frame, stat, (20, y_status + 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if status == 'violation':
        cv2.putText(frame, f"⏱️ Timer: {int(time.time() - last_notification[buffer_key])}s/{jeda_notifikasi}s",
                   (w - 250, y_status + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return frame, {'status': status, 'results': results, 'status_text': status_text}

def add_log(kamera, message, status):
    log_entry = {
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'message': message,
        'status': status
    }
    if kamera == 'internal':
        logs_data['internal'].insert(0, log_entry)
        if len(logs_data['internal']) > MAX_LOGS:
            logs_data['internal'].pop()
    else:
        logs_data['external'].insert(0, log_entry)
        if len(logs_data['external']) > MAX_LOGS:
            logs_data['external'].pop()

# =====================================================================
# 8. GENERATE FRAME UNTUK WEB
# =====================================================================

def generate_frames():
    """Generator untuk streaming video"""
    cap_internal = None
    cap_external = None
    
    # Inisialisasi kamera
    cap_internal = cv2.VideoCapture(1)
    if cap_internal.isOpened():
        cap_internal.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_internal.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera_state['internal_active'] = True
        print("✅ Kamera Internal aktif")
    else:
        camera_state['internal_active'] = False
        print("❌ Kamera Internal gagal")
    
    # Coba kamera eksternal
    for index in [0, 2]:
        cap_external = cv2.VideoCapture(index)
        if cap_external.isOpened():
            cap_external.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap_external.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera_state['external_active'] = True
            print(f"✅ Kamera Eksternal (index {index}) aktif")
            break
        else:
            cap_external = None
    
    if cap_external is None:
        camera_state['external_active'] = False
        print("⚠️ Kamera Eksternal tidak ditemukan")
    
    camera_state['running'] = True
    
    while camera_state['running']:
        frame_internal = None
        frame_external = None
        results_internal = None
        results_external = None
        
        # Proses frame internal
        if cap_internal is not None and cap_internal.isOpened():
            ret_internal, frame_internal_raw = cap_internal.read()
            if ret_internal and frame_internal_raw is not None:
                frame_internal, results_internal = process_frame(frame_internal_raw, "KAMERA INTERNAL", 'internal')
                latest_frames['internal'] = frame_internal
                
                # Cek pelanggaran internal
                if results_internal and results_internal['status'] == 'violation':
                    waktu_sekarang = time.time()
                    if waktu_sekarang - last_notification['internal'] > jeda_notifikasi:
                        last_notification['internal'] = waktu_sekarang
                        
                        now = datetime.now()
                        file_time = now.strftime("%Y%m%d_%H%M%S")
                        waktu_str = now.strftime("%Y-%m-%d %H:%M:%S")
                        
                        nama_file = f"pelanggar_internal_{file_time}.jpg"
                        full_path = os.path.join(output_folder, nama_file)
                        
                        cv2.imwrite(full_path, frame_internal)
                        
                        detail_json = f"Manusia: {len(results_internal['results']['human'])}, Rokok: {len(results_internal['results']['rokok'])}, Asap: {len(results_internal['results']['asap'])}"
                        cursor.execute('''
                            INSERT INTO logs (waktu, jenis_pelanggaran, kamera_sumber, lokasi_foto, detail_deteksi)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (waktu_str, "MEROKOK DEKAT MULUT", "Kamera Internal", f"violators/{nama_file}", detail_json))
                        conn.commit()
                        
                        pesan = create_telegram_message(results_internal['results'], results_internal['status'], results_internal['status_text'], "Kamera Internal")
                        kirim_notifikasi_telegram(full_path, pesan, "Kamera Internal")
                        
                        add_log('internal', f"🚨 PELANGGARAN: {results_internal['status_text']}", 'danger')
        
        # Proses frame eksternal
        if cap_external is not None and cap_external.isOpened():
            ret_external, frame_external_raw = cap_external.read()
            if ret_external and frame_external_raw is not None:
                frame_external, results_external = process_frame(frame_external_raw, "KAMERA EKSTERNAL", 'external')
                latest_frames['external'] = frame_external
                
                if results_external and results_external['status'] == 'violation':
                    waktu_sekarang = time.time()
                    if waktu_sekarang - last_notification['external'] > jeda_notifikasi:
                        last_notification['external'] = waktu_sekarang
                        
                        now = datetime.now()
                        file_time = now.strftime("%Y%m%d_%H%M%S")
                        waktu_str = now.strftime("%Y-%m-%d %H:%M:%S")
                        
                        nama_file = f"pelanggar_eksternal_{file_time}.jpg"
                        full_path = os.path.join(output_folder, nama_file)
                        
                        cv2.imwrite(full_path, frame_external)
                        
                        detail_json = f"Manusia: {len(results_external['results']['human'])}, Rokok: {len(results_external['results']['rokok'])}, Asap: {len(results_external['results']['asap'])}"
                        cursor.execute('''
                            INSERT INTO logs (waktu, jenis_pelanggaran, kamera_sumber, lokasi_foto, detail_deteksi)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (waktu_str, "MEROKOK DEKAT MULUT", "Kamera Eksternal", f"violators/{nama_file}", detail_json))
                        conn.commit()
                        
                        pesan = create_telegram_message(results_external['results'], results_external['status'], results_external['status_text'], "Kamera Eksternal")
                        kirim_notifikasi_telegram(full_path, pesan, "Kamera Eksternal")
                        
                        add_log('external', f"🚨 PELANGGARAN: {results_external['status_text']}", 'danger')
        
        # Gabungkan frame untuk ditampilkan
        if camera_state['mode'] == 'single':
            # Tampilkan kamera yang dipilih
            selected = camera_state.get('selected_camera', 'internal')
            if selected == 'internal' and frame_internal is not None:
                display_frame = cv2.resize(frame_internal, (960, 540))
            elif selected == 'external' and frame_external is not None:
                display_frame = cv2.resize(frame_external, (960, 540))
            elif frame_internal is not None:
                display_frame = cv2.resize(frame_internal, (960, 540))
            elif frame_external is not None:
                display_frame = cv2.resize(frame_external, (960, 540))
            else:
                display_frame = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(display_frame, "Tidak ada kamera aktif", (300, 270),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Tampilkan dual camera
            if frame_internal is not None and frame_external is not None:
                frame_internal_resized = cv2.resize(frame_internal, (480, 360))
                frame_external_resized = cv2.resize(frame_external, (480, 360))
                display_frame = np.hstack([frame_internal_resized, frame_external_resized])
            elif frame_internal is not None:
                display_frame = cv2.resize(frame_internal, (960, 540))
            elif frame_external is not None:
                display_frame = cv2.resize(frame_external, (960, 540))
            else:
                display_frame = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(display_frame, "Tidak ada kamera aktif", (300, 270),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Cleanup
    if cap_internal is not None:
        cap_internal.release()
    if cap_external is not None:
        cap_external.release()

# =====================================================================
# 9. ROUTES
# =====================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logs')
def logs_page():
    return render_template('logs.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json
    mode = data.get('mode', 'single')
    selected = data.get('selected_camera', 'internal')
    camera_state['mode'] = mode
    camera_state['selected_camera'] = selected
    return jsonify({'status': 'success', 'mode': mode, 'selected': selected})

@app.route('/get_logs')
def get_logs():
    return jsonify({
        'internal': logs_data['internal'],
        'external': logs_data['external']
    })

@app.route('/get_stats')
def get_stats():
    # Hitung total pelanggaran dari database
    cursor.execute('SELECT COUNT(*) FROM logs')
    total_violations = cursor.fetchone()[0]

    return jsonify({
        'internal': current_stats['internal'],
        'external': current_stats['external'],
        'total_violations': total_violations
    })


@app.route('/get_logs_db')
def get_logs_db():
    cursor.execute('SELECT * FROM logs ORDER BY id DESC LIMIT 100')
    logs = cursor.fetchall()
    return jsonify([{
        'id': log[0],
        'waktu': log[1],
        'jenis': log[2],
        'kamera': log[3],
        'foto': log[4],
        'detail': log[5]
    } for log in logs])

@app.route('/image/<path:filename>')
def get_image(filename):
    return send_file(os.path.join(output_folder, filename))

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    global logs_data
    data = request.json
    kamera = data.get('kamera', 'all')
    if kamera == 'all':
        logs_data['internal'] = []
        logs_data['external'] = []
    elif kamera == 'internal':
        logs_data['internal'] = []
    elif kamera == 'external':
        logs_data['external'] = []
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)