# 🚭 SMOKING DETECTION SYSTEM WITH YOLOv8

## 📌 Deskripsi Project

Smoking Detection System adalah sistem berbasis Artificial Intelligence (AI) dan Computer Vision yang digunakan untuk mendeteksi aktivitas merokok secara realtime menggunakan webcam atau kamera CCTV.

Sistem ini dibangun menggunakan:

- YOLOv8
- Python
- OpenCV
- SQLite Database
- Telegram Bot API

Project ini mampu:

✅ Mendeteksi rokok dan vape secara realtime  
✅ Menyimpan foto pelanggar otomatis  
✅ Menyimpan histori pelanggaran ke database  
✅ Mengirim notifikasi Telegram otomatis  
✅ Mendukung single webcam dan dual webcam  
✅ Digunakan untuk monitoring area larangan merokok

---

# 🎯 Tujuan Project

Tujuan dari project ini adalah:

- Membantu pengawasan area bebas rokok secara otomatis
- Mengurangi pelanggaran merokok di lingkungan kampus atau instansi
- Mengembangkan implementasi Computer Vision berbasis YOLOv8
- Menjadi sistem monitoring realtime berbasis AI

---

# 🧠 Teknologi yang Digunakan

| Teknologi | Fungsi |
|---|---|
| Python | Bahasa pemrograman utama |
| YOLOv8 | Model object detection |
| OpenCV | Pengolahan citra dan webcam |
| SQLite | Database penyimpanan pelanggaran |
| Telegram Bot API | Notifikasi realtime |
| Ultralytics | Framework YOLOv8 |

---

# 📂 Struktur Folder Project

```bash
SMOKING_DETECTION/
│
├── cigaret_dataset/              # Dataset rokok
├── vape_dataset/                 # Dataset vape
├── dataset/                      # Dataset gabungan
│
├── models/
│   └── best.pt                   # Model hasil training terbaik
│
├── runs/
│   └── detect/
│       ├── train/
│       └── train-2/
│
├── src/
│   ├── violators/                # Folder penyimpanan foto pelanggar
│   ├── baca_db.py                # Melihat isi database
│   ├── detect_dual_webcam.py     # Deteksi dual webcam
│   ├── detect_webcam.py          # Deteksi realtime webcam
│   ├── log_pelanggaran.db        # Database SQLite
│   └── train.py                  # Training model YOLOv8
│
├── change_vape_label.py          # Mengubah label dataset vape
├── merge_dataset.py              # Menggabungkan dataset
├── dataset.zip
├── requirements.txt
├── yolov8n.pt
└── README.md
```

---

# 🖥️ Spesifikasi Minimum

## Hardware

- Processor Intel i5 / Ryzen 5
- RAM minimal 8GB
- Webcam / CCTV
- GPU NVIDIA (opsional tetapi direkomendasikan)

## Software

- Windows 10/11
- Python 3.10+
- Visual Studio Code
- Git

---

# ⚙️ Cara Instalasi

## 1. Clone Repository

```bash
git clone https://github.com/username/smoking_detection.git
cd smoking_detection
```

---

## 2. Membuat Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Semua Dependency

```bash
pip install -r requirements.txt
```

---

# 📦 Isi requirements.txt

```txt
ultralytics
opencv-python
numpy
sqlite3
python-telegram-bot
```

---

# 📊 Dataset

Dataset yang digunakan terdiri dari:

- Cigarette
- Vape

Format dataset:

```bash
images/
labels/
```

Dataset kemudian digabung menggunakan:

```bash
python merge_dataset.py
```

---

# 🔄 Mengubah Label Vape

Jika label vape berbeda dengan cigarette, jalankan:

```bash
python change_vape_label.py
```

Script ini digunakan agar semua label sesuai dengan class YOLO.

---

# 🧠 Training Model YOLOv8

## Menjalankan Training

```bash
python src/train.py
```

Atau menggunakan command langsung:

```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

---

# 📁 Hasil Training

Model hasil training akan tersimpan di:

```bash
runs/detect/train/weights/best.pt
```

---

# 🎥 Menjalankan Deteksi Realtime

## Single Webcam

```bash
python src/detect_webcam.py
```

---

## Dual Webcam

```bash
python src/detect_dual_webcam.py
```

---

# 📸 Sistem Penyimpanan Foto Pelanggar

Ketika sistem mendeteksi aktivitas merokok:

- Foto otomatis disimpan
- Nama file berdasarkan waktu deteksi
- Tersimpan di folder:

```bash
src/violators/
```

Contoh:

```bash
pelanggar_smoking_20260525_133408.jpg
```

---

# 🗄️ Database Pelanggaran

Database menggunakan SQLite.

File database:

```bash
src/log_pelanggaran.db
```

Data yang disimpan:

- ID
- Jenis pelanggaran
- Waktu pelanggaran
- Lokasi foto

---

# 📖 Membaca Isi Database

Jalankan:

```bash
python src/baca_db.py
```

---

# 📲 Integrasi Telegram Bot

Sistem dapat mengirim notifikasi otomatis ketika pelanggaran terdeteksi.

## Contoh Notifikasi

```text
⚠️ Pelanggaran Terdeteksi!
Jenis: Smoking
Waktu: 2026-05-25 13:34:08
```

---

# 🤖 Cara Membuat Telegram Bot

## 1. Buka Telegram

Cari:

```text
@BotFather
```

---

## 2. Buat Bot Baru

Gunakan command:

```text
/newbot
```

---

## 3. Simpan Token Bot

Contoh:

```text
123456789:AAxxxxxxxxxxxxxxxxxxxx
```

---

# 🔍 Alur Sistem

```text
Kamera Webcam
       ↓
Frame Realtime
       ↓
YOLOv8 Detection
       ↓
Deteksi Rokok/Vape
       ↓
Simpan Foto
       ↓
Simpan Database
       ↓
Kirim Telegram
```

---

# 🧪 Pengujian Sistem

Pengujian dilakukan menggunakan:

- Webcam laptop
- Kamera USB eksternal
- Berbagai kondisi pencahayaan
- Deteksi cigarette dan vape

---

# 📈 Pengembangan Selanjutnya

Fitur yang dapat ditambahkan:

- Dashboard monitoring web
- Heatmap area pelanggaran
- Rekap statistik harian
- Multi camera CCTV
- Export laporan PDF
- Sistem Android monitoring
- Face recognition pelanggar
- GPS lokasi kamera
- Cloud database

---

# 🛠️ Troubleshooting

## Webcam Tidak Terdeteksi

Coba ubah index kamera:

```python
cap = cv2.VideoCapture(1)
```

---

## Module Not Found

Install ulang dependency:

```bash
pip install -r requirements.txt
```

---

## CUDA Tidak Aktif

Cek GPU:

```python
import torch
print(torch.cuda.is_available())
```

---

# 📚 Referensi

- YOLOv8 Ultralytics
- OpenCV Documentation
- Python Official Documentation
- SQLite Documentation

---

# 👨‍💻 Author

Developed by:

Rani  
Teknik Informatika

Project Skripsi Artificial Intelligence & Computer Vision

---

# 📄 License

Project ini dibuat untuk keperluan:

- Pendidikan
- Penelitian
- Pengembangan AI

Non-commercial use only.

