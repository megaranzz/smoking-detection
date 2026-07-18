# clean_project_fixed.py
import os
import shutil
import sqlite3
import time

print("🧹 MEMBERSIHKAN PROYEK...")
print("="*50)

# =====================================================================
# FUNGSI UNTUK HAPUS FOLDER DENGAN RETRY
# =====================================================================

def hapus_folder_dengan_retry(path, max_retry=3):
    """Hapus folder dengan beberapa percobaan"""
    for i in range(max_retry):
        try:
            if os.path.exists(path):
                # Ubah atribut file agar bisa dihapus
                for root, dirs, files in os.walk(path):
                    for file in files:
                        try:
                            os.chmod(os.path.join(root, file), 0o777)
                        except:
                            pass
                    for dir in dirs:
                        try:
                            os.chmod(os.path.join(root, dir), 0o777)
                        except:
                            pass
                
                shutil.rmtree(path)
                print(f"🗑️  Folder dihapus: {path}/")
                return True
            else:
                print(f"ℹ️  Folder tidak ditemukan: {path}/")
                return True
        except PermissionError as e:
            print(f"⚠️  Percobaan {i+1}/{max_retry}: {path} - {e}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Gagal hapus {path}: {e}")
            break
    
    print(f"❌ Tidak bisa hapus {path} - coba hapus manual")
    return False

# =====================================================================
# 1. HAPUS FILE/FOLDER TIDAK PENTING
# =====================================================================

print("\n📁 MENGHAPUS FILE/FOLDER TIDAK PENTING:")
print("-"*50)

# Folder yang akan dihapus
folders_to_delete = [
    'cigar_dataset',
    'vape_dataset', 
    'dataset',
    'runs',
    'violators_hp',
    'yolov8n_openvino_model'
]

# File yang akan dihapus
files_to_delete = [
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

# Hapus folder dengan retry
for folder in folders_to_delete:
    hapus_folder_dengan_retry(folder)

# Hapus file
for file in files_to_delete:
    if os.path.exists(file):
        try:
            os.remove(file)
            print(f"🗑️  File dihapus: {file}")
        except Exception as e:
            print(f"❌ Gagal hapus {file}: {e}")
    else:
        print(f"ℹ️  File tidak ditemukan: {file}")

# =====================================================================
# 2. RESET DATABASE (log_pelanggaran.db)
# =====================================================================

print("\n💾 RESET DATABASE (log_pelanggaran.db):")
print("-"*50)

db_path = 'log_pelanggaran.db'

if os.path.exists(db_path):
    try:
        os.remove(db_path)
        print(f"🗑️  Database dihapus: {db_path}")
        print("ℹ️  Database akan dibuat ulang otomatis saat program dijalankan")
    except Exception as e:
        print(f"❌ Gagal hapus database: {e}")
        
        # Coba kosongkan isinya
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM logs")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='logs'")
            conn.commit()
            conn.close()
            print(f"✅ Isi database dikosongkan")
        except Exception as e2:
            print(f"❌ Gagal kosongkan database: {e2}")
else:
    print(f"ℹ️  Database tidak ditemukan (akan dibuat otomatis)")

# =====================================================================
# 3. RESET FOLDER VIOLATORS
# =====================================================================

print("\n📸 RESET FOLDER VIOLATORS:")
print("-"*50)

violators_path = 'violators'

if os.path.exists(violators_path) and os.path.isdir(violators_path):
    try:
        files = os.listdir(violators_path)
        file_count = len(files)
        
        for file in files:
            file_path = os.path.join(violators_path, file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        
        print(f"🗑️  {file_count} file foto dihapus dari {violators_path}/")
    except Exception as e:
        print(f"❌ Gagal hapus file: {e}")
else:
    try:
        os.makedirs(violators_path)
        print(f"✅ Folder {violators_path}/ dibuat")
    except:
        pass

# =====================================================================
# 4. CLEAN SRC FOLDER (TANPA ERROR)
# =====================================================================

print("\n📂 CLEAN FOLDER SRC (Opsional):")
print("-"*50)

src_path = 'src'
if os.path.exists(src_path) and os.path.isdir(src_path):
    src_files = os.listdir(src_path)
    if src_files:
        print(f"ℹ️  Folder src/ berisi {len(src_files)} file:")
        for f in src_files[:5]:
            print(f"   - {f}")
        if len(src_files) > 5:
            print(f"   ... dan {len(src_files)-5} file lainnya")
        
        konfirmasi = input("\n❓ Hapus semua isi folder src/? (y/n): ")
        if konfirmasi.lower() == 'y':
            for item in src_files:
                item_path = os.path.join(src_path, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        print(f"   🗑️  File: {item}")
                    elif os.path.isdir(item_path):
                        # Coba hapus folder dengan retry
                        hapus_folder_dengan_retry(item_path)
                except Exception as e:
                    print(f"   ⚠️  Tidak bisa hapus {item}: {e}")
            
            print(f"✅ Pembersihan src/ selesai")
        else:
            print("ℹ️  Folder src/ tidak dihapus")
    else:
        print("ℹ️  Folder src/ kosong")

# =====================================================================
# 5. RINGKASAN
# =====================================================================

print("\n" + "="*50)
print("✅ PEMBERSIHAN SELESAI!")
print("="*50)

print("\n📊 STATUS:")
print(f"   - Database: {'DIHAPUS' if not os.path.exists(db_path) else 'TETAP ADA'}")
print(f"   - Violators: {'KOSONG' if os.path.exists(violators_path) else 'TIDAK ADA'}")

print("\n💡 CATATAN:")
print("   - Jika ada folder yang tidak bisa dihapus, tutup program yang menggunakan folder tersebut")
print("   - Folder yang tidak terhapus bisa dihapus manual melalui Windows Explorer")

print("\n🚀 Proyek siap dijalankan!")
print("   Jalankan: python app.py")