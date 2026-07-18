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

def kirim_notifikasi_telegram(foto_path, pesan_teks):
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
# 3. FUNGSI TEST TELEGRAM
# =====================================================================

def test_koneksi_telegram():
    print("\n" + "="*50)
    print("🔍 TESTING KONEKSI TELEGRAM")
    print("="*50)
    
    success_count = 0
    
    for idx, bot in enumerate(TELEGRAM_BOTS, 1):
        try:
            token = bot["token"]
            chat_id = bot["chat_id"]
            
            url = f"https://api.telegram.org/bot{token}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    username = data.get('result', {}).get('username', 'unknown')
                    print(f"✅ Bot {idx}: @{username} AKTIF")
                    success_count += 1
                    
                    test_url = f"https://api.telegram.org/bot{token}/sendMessage"
                    test_data = {
                        'chat_id': chat_id,
                        'text': '🔔 TEST: Smart CCTV System Connected!'
                    }
                    test_response = requests.post(test_url, data=test_data, timeout=10)
                    
                    if test_response.status_code == 200:
                        print(f"   ✅ Test message terkirim")
                    else:
                        print(f"   ⚠️ Test message gagal")
                else:
                    print(f"❌ Bot {idx}: Tidak aktif")
            else:
                print(f"❌ Bot {idx}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Bot {idx}: Error - {e}")
    
    print("\n" + "="*50)
    print(f"📊 Hasil: {success_count}/{len(TELEGRAM_BOTS)} bot aktif")
    print("="*50 + "\n")
    
    return success_count > 0

telegram_ready = test_koneksi_telegram()

# =====================================================================
# 4. SETUP FOLDER & DATABASE
# =====================================================================

output_folder = os.path.join(os.path.dirname(__file__), 'violators')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

db_path = os.path.join(os.path.dirname(__file__), 'log_pelanggaran.db')
conn = sqlite3.connect(db_path)
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
# 5. LOAD MODEL YOLO
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
# 6. MEDIAPIPE FACE MESH
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
# 7. KONFIGURASI
# =====================================================================

YOLO_CONF_OBJEK = 0.35
YOLO_CONF_ASAP = 0.30
YOLO_CONF_HUMAN = 0.30
YOLO_IMGSZ = 640

# Jarak rokok ke mulut (pixel)
MOUTH_DISTANCE_THRESHOLD = 150

# Buffer untuk smoothing
DETECTION_BUFFER = 5
human_buffer = deque(maxlen=DETECTION_BUFFER)
smoke_buffer = deque(maxlen=DETECTION_BUFFER)
objek_buffer = deque(maxlen=DETECTION_BUFFER)

# Timer notifikasi
terakhir_notif = 0
jeda_notifikasi = 10

# FPS
frame_count = 0
fps_start = time.time()
fps_count = 0
current_fps = 0

print("\n🟢 SISTEM DETEKSI PELANGGARAN ROKOK")
print("🔹 Logika Deteksi:")
print("   1. Orang Terdeteksi → ✅ Normal")
print("   2. Orang + Asap → ⚠️ Peringatan (tidak disimpan)")
print("   3. Orang + Rokok/Vape → ⚠️ Peringatan (tidak disimpan)")
print("   4. Orang + Rokok/Vape + Asap → 🚨 PELANGGARAN (disimpan + Telegram)")
print(f"🔹 Confidence Rokok: {YOLO_CONF_OBJEK}")
print(f"🔹 Confidence Asap: {YOLO_CONF_ASAP}")
print(f"🔹 Jarak Rokok ke Mulut: {MOUTH_DISTANCE_THRESHOLD}px")
print("🔹 Tekan 'q' untuk keluar\n")

# =====================================================================
# 8. BUKA KAMERA
# =====================================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Gagal membuka kamera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("✅ Kamera berhasil diakses!")

# =====================================================================
# 9. FUNGSI DETEKSI DENGAN FACEMESH
# =====================================================================

def get_mouth_position(roi):
    """Deteksi posisi mulut dari ROI menggunakan FaceMesh."""
    try:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = roi.shape[:2]
            
            # Titik mulut yang lebih akurat
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
    """Cek apakah rokok dekat dengan mulut menggunakan FaceMesh."""
    px1, py1, px2, py2 = person_bbox
    
    # Ambil ROI area kepala (40% dari bounding box)
    head_y2 = py1 + int((py2 - py1) * 0.4)
    roi_head = frame[py1:head_y2, px1:px2]
    
    if roi_head.size == 0:
        return False, 999
    
    # Deteksi posisi mulut dengan FaceMesh
    mouth_pos = get_mouth_position(roi_head)
    
    if mouth_pos is None:
        return False, 999
    
    # Konversi posisi mulut ke koordinat global
    mouth_global = (px1 + mouth_pos[0], py1 + mouth_pos[1])
    
    # Hitung jarak rokok ke mulut
    sx, sy = smoking_center
    distance = math.sqrt((sx - mouth_global[0])**2 + (sy - mouth_global[1])**2)
    
    # Gambar titik mulut (HANYA WARNA, TANPA TEKS)
    cv2.circle(frame, mouth_global, 8, (0, 255, 0), -1)
    cv2.circle(frame, mouth_global, 10, (0, 255, 0), 1)
    
    # Gambar garis ke rokok jika dekat
    if distance < MOUTH_DISTANCE_THRESHOLD:
        cv2.line(frame, mouth_global, smoking_center, (0, 0, 255), 2)
        mid_x = (mouth_global[0] + smoking_center[0]) // 2
        mid_y = (mouth_global[1] + smoking_center[1]) // 2
        cv2.putText(frame, f"DEKAT {distance:.0f}px", (mid_x-30, mid_y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return distance < MOUTH_DISTANCE_THRESHOLD, distance

def detect_all_objects_with_mouth(frame):
    """Deteksi semua objek dengan deteksi mulut"""
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
    
    # === CEK ROKOK DEKAT MULUT UNTUK SETIAP ORANG ===
    for person in results['human']:
        person_bbox = person['bbox']
        person['near_mouth'] = False
        person['near_mouth_distance'] = 999
        person['has_rokok'] = False
        person['has_asap'] = False
        
        # Cek rokok dekat mulut
        for rokok in results['rokok']:
            is_near, dist = is_smoking_near_mouth(rokok['center'], person_bbox, frame)
            if is_near:
                person['near_mouth'] = True
                person['near_mouth_distance'] = dist
                person['has_rokok'] = True
                break
        
        # Cek asap di sekitar orang
        for asap in results['asap']:
            ax, ay = asap['center']
            px1, py1, px2, py2 = person_bbox
            padding = 50
            if (px1 - padding < ax < px2 + padding) and (py1 - padding < ay < py2 + padding):
                person['has_asap'] = True
                break
    
    return results

def draw_detections_with_id(frame, results):
    """Gambar semua deteksi dengan ID orang"""
    COLORS = {
        'human': (255, 0, 0),      # Biru
        'rokok': (0, 0, 255),       # Merah
        'asap': (0, 255, 255)       # Kuning
    }
    
    # Gambar Manusia dengan ID
    for det in results['human']:
        x1, y1, x2, y2 = det['bbox']
        conf = det['conf']
        person_id = det.get('id', 0)
        near_mouth = det.get('near_mouth', False)
        
        # Warna bounding box berdasarkan status
        if near_mouth:
            color = (0, 0, 255)  # Merah jika rokok dekat mulut
        else:
            color = COLORS['human']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Label dengan ID
        label = f"ORANG #{person_id} {conf:.2f}"
        if near_mouth:
            label += " 🔥"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Tampilkan status rokok dekat mulut
        if near_mouth:
            cv2.putText(frame, "ROKOK DEKAT MULUT!", (x1, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Gambar Rokok
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
        
        # Titik tengah
        cx, cy = det['center']
        cv2.circle(frame, (cx, cy), 5, COLORS['rokok'], -1)
    
    # Gambar Asap
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
    """Cek level pelanggaran dengan deteksi mulut"""
    has_human = len(results['human']) > 0
    has_rokok = len(results['rokok']) > 0
    has_asap = len(results['asap']) > 0
    
    # Cek apakah ada orang dengan rokok dekat mulut
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

def create_telegram_message_with_mouth(results, status, status_text):
    """Buat pesan Telegram dengan detail deteksi - FORMAT SAMA PERSIS"""
    now = datetime.now()
    waktu_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    total_human = len(results['human'])
    total_rokok = len(results['rokok'])
    total_asap = len(results['asap'])
    
    # Format pesan dengan emoji yang sama seperti sebelumnya
    message = f"🔔 [SMOKING DETECTED] 🔔\n"
    message += f"━━━━━━━━━━━━━━━━━━━\n"
    message += f"🚨 {status_text}\n"
    message += f"━━━━━━━━━━━━━━━━━━━\n"
    message += f"📸 Kamera: Kamera Internal\n"
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
            
            # Cari rokok terdekat untuk orang ini
            closest_rokok_conf = 0
            for rokok in results['rokok']:
                hx, hy = person['center']
                rx, ry = rokok['center']
                distance = math.sqrt((hx - rx)**2 + (hy - ry)**2)
                if distance < 300:
                    if rokok['conf'] > closest_rokok_conf:
                        closest_rokok_conf = rokok['conf']
            
            # Cari asap terdekat untuk orang ini
            closest_asap_conf = 0
            for asap in results['asap']:
                hx, hy = person['center']
                ax, ay = asap['center']
                distance = math.sqrt((hx - ax)**2 + (hy - ay)**2)
                if distance < 300:
                    if asap['conf'] > closest_asap_conf:
                        closest_asap_conf = asap['conf']
            
            # Status Rokok
            if has_rokok and near_mouth:
                message += f"   • Rokok/Vape: ✅ TERDETEKSI DEKAT MULUT!\n"
                message += f"     (Confidence: {closest_rokok_conf:.2f}, Jarak: {near_distance:.0f}px)\n"
            elif has_rokok:
                message += f"   • Rokok/Vape: ✅ TERDETEKSI!\n"
                message += f"     (Confidence: {closest_rokok_conf:.2f})\n"
            else:
                message += f"   • Rokok/Vape: ❌ Tidak terdeteksi\n"
            
            # Status Asap
            if has_asap:
                message += f"   • Asap: ✅ TERDETEKSI!\n"
                message += f"     (Confidence: {closest_asap_conf:.2f})\n"
            else:
                message += f"   • Asap: ❌ Tidak terdeteksi\n"
            
            # Status jarak ke mulut
            if near_mouth:
                message += f"   • ✅ Jarak ke mulut: {near_distance:.0f}px (DEKAT!)\n"
            else:
                message += f"   • ✅ Jauh dari mulut\n"
    
    message += f"\n━━━━━━━━━━━━━━━━━━━\n"
    message += f"📎 Lampiran foto terlampir\n"
    message += f"⚙️ Confidence Threshold:\n"
    message += f"   • Rokok: {YOLO_CONF_OBJEK}\n"
    message += f"   • Asap: {YOLO_CONF_ASAP}\n"
    message += f"   • Jarak mulut: {MOUTH_DISTANCE_THRESHOLD}px"
    
    return message

# =====================================================================
# 10. MAIN LOOP
# =====================================================================

print("🔄 Memulai deteksi...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal mengambil gambar.")
        break
    
    frame_count += 1
    h, w, _ = frame.shape
    
    # FPS
    fps_count += 1
    if time.time() - fps_start >= 1.0:
        current_fps = fps_count
        fps_count = 0
        fps_start = time.time()
    
    # === DETEKSI DENGAN FACEMESH ===
    results = detect_all_objects_with_mouth(frame)
    
    # === GAMBAR BOUNDING BOX DENGAN ID ===
    frame = draw_detections_with_id(frame, results)
    
    # === CEK PELANGGARAN ===
    status, status_text = check_violation_with_mouth(results)
    
    # === UPDATE BUFFER ===
    has_human = len(results['human']) > 0
    has_rokok = len(results['rokok']) > 0
    has_asap = len(results['asap']) > 0
    
    human_buffer.append(has_human)
    smoke_buffer.append(has_asap)
    objek_buffer.append(has_rokok)
    
    # === TAMPILAN STATUS ===
    # Info panel
    info_bg = np.zeros((200, 400, 3), dtype=np.uint8)
    info_bg[:] = (0, 0, 0)
    info_bg = cv2.addWeighted(info_bg, 0.5, frame[10:210, 10:410], 0.5, 0)
    frame[10:210, 10:410] = info_bg
    
    y_pos = 30
    cv2.putText(frame, "📹 CCTV MONITOR", (20, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {current_fps}", (w - 100, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Status
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
    
    # Statistik
    stats = [
        f"👤 Manusia: {len(results['human'])}",
        f"🚬 Rokok: {len(results['rokok'])}",
        f"💨 Asap: {len(results['asap'])}",
        f"📊 Total: {len(results['human']) + len(results['rokok']) + len(results['asap'])}"
    ]
    
    for i, stat in enumerate(stats):
        cv2.putText(frame, stat, (20, y_status + 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Timer notifikasi
    if status == 'violation':
        cv2.putText(frame, f"⏱️ Timer: {int(time.time() - terakhir_notif)}s/{jeda_notifikasi}s",
                   (w - 250, y_status + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # === LOGIKA PELANGGARAN ===
    if status == 'violation':
        waktu_sekarang = time.time()
        
        # Overlay merah
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        cv2.putText(frame, "🚨 PELANGGARAN!", (w//2 - 150, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        if waktu_sekarang - terakhir_notif > jeda_notifikasi:
            terakhir_notif = waktu_sekarang
            
            # Simpan ke database
            now = datetime.now()
            file_time = now.strftime("%Y%m%d_%H%M%S")
            waktu_str = now.strftime("%Y-%m-%d %H:%M:%S")
            
            nama_file = f"pelanggar_{file_time}.jpg"
            full_path = os.path.join(output_folder, nama_file)
            
            # Simpan foto
            cv2.imwrite(full_path, frame)
            print(f"📸 Foto disimpan: {full_path}")
            
            # Simpan ke database
            detail_json = f"Manusia: {len(results['human'])}, Rokok: {len(results['rokok'])}, Asap: {len(results['asap'])}"
            cursor.execute('''
                INSERT INTO logs (waktu, jenis_pelanggaran, kamera_sumber, lokasi_foto, detail_deteksi)
                VALUES (?, ?, ?, ?, ?)
            ''', (waktu_str, "MEROKOK DEKAT MULUT", "Webcam", f"violators/{nama_file}", detail_json))
            conn.commit()
            print("💾 Database tersimpan")
            
            # Kirim Telegram
            pesan = create_telegram_message_with_mouth(results, status, status_text)
            kirim_notifikasi_telegram(full_path, pesan)
            print("📤 Notifikasi Telegram dikirim!")
    
    elif status in ['warning_asap', 'warning_rokok']:
        # Peringatan di layar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.putText(frame, status_text, (w//2 - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # === TAMPILKAN FRAME ===
    display_frame = cv2.resize(frame, (960, 540))
    cv2.imshow('Sistem Deteksi Pelanggaran Rokok - Kampus', display_frame)
    
    # === KEYBOARD ===
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        terakhir_notif = 0
        print("🔄 Reset timer notifikasi")

# =====================================================================
# CLEANUP
# =====================================================================

conn.close()
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("✅ Program selesai")
print(f"📊 Total frame diproses: {frame_count}")
print("="*50)