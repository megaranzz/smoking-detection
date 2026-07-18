# cek_struktur_final.py
import os
import sys
import sqlite3

print("="*60)
print("🔍 CEK STRUKTUR PROYEK FINAL")
print("="*60)
print(f"📂 Direktori: {os.getcwd()}")
print("="*60)

# =====================================================================
# 1. CEK FOLDER PENTING
# =====================================================================

print("\n📁 CEK FOLDER:")
print("-"*50)

folders = {
    'models': 'Model YOLO (WAJIB)',
    'templates': 'HTML Templates (WAJIB)',
    'venv': 'Virtual Environment',
    'violators': 'Folder Foto Pelanggar'
}

for folder, desc in folders.items():
    if os.path.exists(folder) and os.path.isdir(folder):
        # Hitung isi folder
        if folder == 'models':
            files = os.listdir(folder)
            print(f"✅ {folder}/ - {desc}")
            print(f"   📄 Isi: {', '.join(files) if files else 'kosong'}")
        elif folder == 'templates':
            files = os.listdir(folder)
            print(f"✅ {folder}/ - {desc}")
            print(f"   📄 Isi: {', '.join(files) if files else 'kosong'}")
        elif folder == 'violators':
            files = os.listdir(folder)
            print(f"✅ {folder}/ - {desc}")
            print(f"   📄 Isi: {len(files)} file" if files else "   📄 Isi: kosong")
        else:
            print(f"✅ {folder}/ - {desc}")
    else:
        print(f"❌ {folder}/ - {desc} TIDAK DITEMUKAN!")

# =====================================================================
# 2. CEK FILE PENTING
# =====================================================================

print("\n📄 CEK FILE:")
print("-"*50)

files = {
    'app.py': 'Web Dashboard (WAJIB)',
    'dual_kameranya.py': 'Dual Camera Detection (WAJIB)',
    'test_gemini_code.py': 'Single Camera Detection (WAJIB)',
    'requirements.txt': 'Dependencies (WAJIB)',
    'readme.md': 'Dokumentasi',
    'yolov8n.pt': 'Model Human Detection'
}

for file, desc in files.items():
    if os.path.exists(file) and os.path.isfile(file):
        size = os.path.getsize(file) / 1024  # KB
        if size > 1024:
            size_str = f"{size/1024:.1f} MB"
        else:
            size_str = f"{size:.1f} KB"
        print(f"✅ {file} - {desc} ({size_str})")
    else:
        print(f"❌ {file} - {desc} TIDAK DITEMUKAN!")

# =====================================================================
# 3. CEK MODEL YOLO
# =====================================================================

print("\n🤖 CEK MODEL YOLO:")
print("-"*50)

model_files = [
    'models/cigaret_vape.pt',
    'models/smoke_only.pt'
]

for model in model_files:
    if os.path.exists(model) and os.path.isfile(model):
        size = os.path.getsize(model) / (1024 * 1024)  # MB
        print(f"✅ {model} ({size:.1f} MB)")
    else:
        print(f"❌ {model} TIDAK DITEMUKAN!")

# =====================================================================
# 4. CEK TEMPLATE HTML
# =====================================================================

print("\n🎨 CEK TEMPLATE HTML:")
print("-"*50)

templates = [
    'templates/index.html',
    'templates/logs.html'
]

for template in templates:
    if os.path.exists(template) and os.path.isfile(template):
        size = os.path.getsize(template) / 1024  # KB
        print(f"✅ {template} ({size:.1f} KB)")
    else:
        print(f"❌ {template} TIDAK DITEMUKAN!")

# =====================================================================
# 5. CEK FILE YANG TIDAK PERLU
# =====================================================================

print("\n⚠️  CEK FILE YANG TIDAK PERLU (masih ada?):")
print("-"*50)

unnecessary = [
    'cigar_dataset',
    'vape_dataset',
    'dataset',
    'runs',
    'violators_hp',
    'yolov8n_openvino_model',
    'dataset.zip',
    'merge_dataset.py',
    'change_vape_label.py',
    'export_model.py',
    'cek_kamera_yolo.py',
    'cek_model.py',
    'cek_struktur_folder.py',
    'test_gemini_code_copy.py',
    'test_kamera_hp_gemini.py',
    'yolov8n.py',
    'screenshot_20260703_193104.jpg',
    'apibottlegram.txt',
    'peringatan.wav',
    'log_pelanggaran_hp.db'
]

found = []
for item in unnecessary:
    if os.path.exists(item):
        found.append(item)

if found:
    print("⚠️  File/folder berikut MASIH ADA (bisa dihapus):")
    for item in found:
        print(f"   - {item}")
else:
    print("✅ Tidak ada file/folder tidak perlu!")

# =====================================================================
# 6. CEK DATABASE
# =====================================================================

print("\n💾 CEK DATABASE:")
print("-"*50)

if os.path.exists('log_pelanggaran.db'):
    size = os.path.getsize('log_pelanggaran.db') / 1024
    print(f"✅ Database ditemukan ({size:.1f} KB)")
    
    # Coba baca isi
    try:
        conn = sqlite3.connect('log_pelanggaran.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        if tables:
            print(f"   📊 Tabel: {', '.join([t[0] for t in tables])}")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"   📊 {table[0]}: {count} data")
        conn.close()
    except Exception as e:
        print(f"   ⚠️ Error membaca database: {e}")
else:
    print("ℹ️  Database belum ada (akan dibuat otomatis saat program dijalankan)")

# =====================================================================
# 7. RINGKASAN
# =====================================================================

print("\n" + "="*60)
print("📊 RINGKASAN")
print("="*60)

# Hitung total
total_ok = 0
total_error = 0

# Cek folder wajib
wajib_folder = ['models', 'templates']
for folder in wajib_folder:
    if os.path.exists(folder) and os.path.isdir(folder):
        total_ok += 1
    else:
        total_error += 1

# Cek file wajib
wajib_file = ['app.py', 'dual_kameranya.py', 'test_gemini_code.py', 'requirements.txt']
for file in wajib_file:
    if os.path.exists(file) and os.path.isfile(file):
        total_ok += 1
    else:
        total_error += 1

print(f"\n✅ Komponen penting: {total_ok} ditemukan")
print(f"❌ Komponen hilang: {total_error}")

if total_error == 0:
    print("\n🎉 SEMUA LENGKAP! PROYEK SIAP DIJALANKAN!")
    print("\n🚀 Jalankan:")
    print("   python app.py")
    print("   # atau")
    print("   python dual_kameranya.py")
    print("   # atau")
    print("   python test_gemini_code.py")
else:
    print("\n⚠️ Ada komponen yang hilang. Pastikan semua file penting ada.")

print("\n" + "="*60)