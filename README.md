# 💻 Computer Vision Laboratory v1.0
> **An End-to-End Deep Learning Web Platform for Everyday Object Detection and Custom Model Re-Training Pipelines Built on Top of YOLOv8 & Flask Framework.**

---

## 🔄 Aplikasi Data Pipelines & Alur Kerja Sistem

Proyek ini dibangun dengan mengintegrasikan empat pilar utama rekayasa perangkat lunak visi komputer, yang terhubung dalam satu jalur pipa data (*unified lifecycle data pipelines*) sebagai berikut:


### 🗂️ Penjelasan Rinci Tiap Tahapan Kerja:

#### 1. Tahap 01: Pengumpulan Data Terdistribusi (LAN & Camera Capture)
* **Proses:** Sistem membuka akses penangkapan citra digital baik menggunakan kamera internal laptop maupun perangkat eksternal iPhone melalui fitur *Continuity Camera* (koneksi Bluetooth & Wi-Fi lokal).
* **Fitur Utama:** Terdapat opsi *Auto-Capture Interval* yang secara otomatis mengambil frame kamera setiap 3 detik. Seluruh aset gambar mentah langsung dialirkan secara aman dan disimpan ke dalam direktori server lokal `ip_dataset/`.

#### 2. Tahap 02: Alat Anotasi Kanvas Interaktif (Interactive Labeling Tool)
* **Proses:** Berkas gambar dari `ip_dataset/` dimuat ke dalam kanvas HTML5 interaktif di halaman web. Pengguna dapat memilih kelas objek dan langsung melakukan teknik penandaan (*annotation positioning*).
* **Fitur Utama:** Mendukung dua mode pelabelan mutakhir: **Bounding Box (📦 Box)** untuk deteksi standar, dan **Polygon (🔷 Segmentation Map)** untuk segmentasi objek kompleks. Koordinat titik piksel objek secara otomatis dinormalisasi menjadi skala `[0-1]` sesuai regulasi dataset YOLO.

#### 3. Tahap 03: Pelatihan Ulang Mandiri Latar Belakang (Secure Worker Thread Re-Training)
* **Proses:** Ketika konfigurasi training dikirimkan dalam bentuk JSON (Base Model, Epochs, Batch Size, Image Size), Flask akan menginisiasi *Background Worker Thread* yang terisolasi secara *native* menggunakan API callback resmi Ultralytics.
* **Fitur Utama:** Sistem 100% aman dari celah bahaya serangan *Remote Code Execution (RCE)* karena tidak lagi mengeksekusi script eksternal lewat terminal via `subprocess`. Log iterasi tiap tahapan (*epoch logs*) disiarkan secara real-time ke browser pengguna menggunakan protokol **Server-Sent Events (SSE)**.

#### 4. Tahap 04: Inferensi Real-Time & Pemicu Alarm Multi-Level (Live Inference & Alert Action)
* **Proses:** Model bobot terbaru (`best_new.pt`) dimuat ke dalam sistem monitor utama menggunakan akselerasi grafis Apple Silicon hardware (*Metal Performance Shaders / MPS*). Aliran frame video diprediksi secara instan dengan ambang batas presisi tinggi (*Confidence Threshold > 85%*).
* **Fitur Utama:** Jika objek pelanggaran keamanan terdeteksi (seperti Handphone):
  * Antarmuka web memicu *Glowing Crimson Border Overlay Alert Banner*.
  * Membunyikan alarm audio frekuensi tinggi (*Square wave beep logic*) langsung dari browser via Web Audio API.
  * Server mengunci tangkapan layar bukti otentik pelanggaran ke dalam direktori `alert_file/` dengan penamaan berbasis waktu (*timestamp*).

---

## 🛠️ Arsitektur Teknologi & Library Pendukung
* **Core Engine AI:** Ultralytics YOLOv8 (PyTorch Deep Learning Backend)
* **Akselerasi Grafis:** Metal Performance Shaders (MPS) Apple Silicon Hardware Optimization
* **Backend Web Server:** Python Flask Micro-framework (Thread-safe architecture)
* **Pemrosesan Citra:** OpenCV (Open Source Computer Vision Library)
* **Antarmuka Pengguna:** HTML5 Canvas API, SSE (Server-Sent Events) Streaming Data Protocol
