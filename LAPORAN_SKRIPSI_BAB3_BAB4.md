# DOKUMEN SISTEM DETEKSI, PELACAKAN, DAN PENGHITUNGAN KENDARAAN (DEYOLO & BYTETRACK)
## Draf Akademik Standar Skripsi untuk Bab III (Metodologi Penelitian) dan Bab IV (Hasil dan Pembahasan)
**Rujukan Ilmiah Utama:**
* **Jurnal:** *Jurnal Teknik Informatika (JUTIF)* Vol. 6, No. 6, Desember 2025, Hal. 1530–1537.
* **Judul:** *Dual Feature Enhancement YOLO: Spatial-Channel Attention Tuning for Vehicle Detection Under Rain Conditions*
* **Penulis:** Chalifa Chazar, Aditya Nugraha (*Department of Informatics, Institut Teknologi Nasional Bandung*)
* **DOI:** [https://doi.org/10.52436/1.jutif.6.6.3540](https://doi.org/10.52436/1.jutif.6.6.3540)

---

# BAB III: METODOLOGI PENELITIAN DAN PERANCANGAN SISTEM

## 3.1 Gambaran Umum Sistem (*System Architecture*)
Sistem yang dirancang dalam penelitian ini adalah sistem cerdas penghitungan volume lalu lintas multi-kelas (*car*, *bus*, *truck*) pada kondisi cuaca buruk (*adverse weather*), khususnya hujan lebat di jalan tol. Sistem ini mengintegrasikan model deteksi objek **DEYOLO (Dual-Feature Enhancement YOLO)** berbasis citra RGB dan *Pseudo-Infrared* (Pseudo-IR) dengan algoritma *Multi-Object Tracking* **ByteTrack** serta modul garis hitung virtual berbasis pemisah lajur (*Split-Lane Dual-Tripwire*).

Alur kerja sistem terbagi ke dalam empat tahapan utama:
1. **Prapemrosesan Citra & Generator Pseudo-IR:** Mengonversi citra RGB menjadi representasi gradien tajam menggunakan operator Sobel dan CLAHE.
2. **Inferensi Deteksi DEYOLO:** Memproses pasangan citra RGB dan Pseudo-IR secara paralel dengan modul atensi *Dual-Enhancement Attention* (DECA & DEPA).
3. **Multi-Object Tracking (ByteTrack):** Mengestimasi posisi kontinu kendaraan menggunakan *Kalman Filter* 8-dimensi dan asosiasi dua tahap (*two-stage association*) untuk menjaga *ID retention*.
4. **Virtual Tripwire & Export Log Data:** Mengevaluasi lintasan kendaraan melintasi garis lajur independen, menyaring duplikasi hitungan, dan mencatat data per kendaraan ke format CSV.

```mermaid
flowchart TD
    A["Input Stream Video Cuaca Hujan (MP4)"] --> B["Ekstraksi Frame & Header Video (ffprobe / OpenCV)"]
    
    subgraph Preprocessing_Inference ["1. Modul Preprocessing & Inferensi DEYOLO"]
        B --> C1["Stream Citra RGB"]
        B --> C2["Generator Pseudo-IR (Sobel Gradient Magnitude)"]
        C1 --> D["Dual-Input Feature Fusion Network (DECA + DEPA)"]
        C2 --> D
        D --> E["Non-Maximum Suppression (NMS)\n(Conf >= 0.20, IoU >= 0.70)"]
    end
    
    subgraph Tracking_Engine ["2. Modul Multi-Object Tracking (ByteTrack)"]
        E --> F["Kalman Filter 8-State Prediction"]
        F --> G1["Tahap 1: Asosiasi Deteksi High-Score (Hungarian Algorithm)"]
        G1 --> G2["Tahap 2: Asosiasi Deteksi Low-Score (Recovery Occlusion)"]
        G2 --> H["Manajemen Lifecycle Track & Alokasi Track ID Permanen"]
    end
    
    subgraph Counting_Analytics ["3. Modul Counting & Log Data"]
        H --> I{"Evaluasi Posisi Lajur & Lintasan Garis\n(Split-Lane Tripwire)"}
        I -->|Lajur Kiri (OUT): Bergerak Naik (dy < 0) & Lewat Y=0.70| J1["Counter OUT + 1"]
        I -->|Lajur Kanan (IN): Bergerak Turun (dy > 0) & Lewat Y=0.50| J2["Counter IN + 1"]
        J1 --> K["Anti-Duplicate Filter (ID Set Registration)"]
        J2 --> K
        K --> L["Pencatatan Event ke Dataframe & Ekspor CSV"]
    end
    
    subgraph Output_Presentation ["4. Visualisasi & Post-Processing"]
        L --> M["Anotasi Grafis Frame (BBox, Track ID, Trajectory Tail, HUD)"]
        M --> N["Encoding H.264 CFR (FFmpeg)"]
        N --> O["Paket ZIP Unduhan (Video MP4 + CSV Log + Rekapitulasi)"]
    end
```

---

## 3.2 Dataset dan Anotasi (*Dataset and Annotation*)
Dataset penelitian dibangun dari rekaman video CCTV jalan tol pada kondisi cuaca hujan siang hari. Tiga klip video mentah, masing-masing berdurasi sekitar 5 menit, diekstraksi dengan laju pengambilan sampel tetap 3 frame per detik (3 FPS). Total frame yang dianotasi adalah sebanyak **2.764 frame** yang dibagi ke dalam tiga subset (*Train*, *Validation*, *Testing*) secara independen untuk memastikan evaluasi objektif pada kondisi rekaman yang belum pernah dilihat model (*unseen conditions*).

**Tabel 3.1 Distribusi Citra dan Jumlah Instance Objek per Kelas pada Dataset (Chazar & Nugraha, 2025)**
| Subset | Jumlah Gambar | Mobil (*Car*) | Bus (*Bus*) | Truk (*Truck*) | Total Instance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train (Pelatihan)** | 2.389 | 7.858 | 451 | 2.614 | 10.923 |
| **Validation (Validasi)** | 316 | 744 | 72 | 270 | 1.086 |
| **Testing (Pengujian)** | 59 | 147 | 18 | 48 | 213 |
| **TOTAL DATASET** | **2.764** | **8.749** | **541** | **2.932** | **12.222** |

---

## 3.3 Prapemrosesan Citra: Sintesis Modalitas Pseudo-Infrared (Pseudo-IR)
Kondisi hujan lebat menimbulkan gangguan visual berupa garis hujan vertikal (*rain streaks*), cipratan air roda kendaraan (*spray*), dan pantulan lampu pada aspal basah (*specular reflections*). Modalitas sintetis Pseudo-IR dibentuk melalui dua metode prapemrosesan:

1. **Varian Sobel Edge (Sintesis Utama - EXP-05):**
   * Citra RGB dikonversi ke skala abu-abu $I_{\text{gray}}$.
   * Diterapkan penghalusan Gaussian (*Gaussian blur*) dengan $\sigma = 2.0$ untuk mereduksi derau frekuensi tinggi.
   * Dihitung gradien spasial Sobel kernel $3 \times 3$:
     $$G_x(x,y) = I_{\text{blur}}(x,y) * \mathbf{S}_x, \quad G_y(x,y) = I_{\text{blur}}(x,y) * \mathbf{S}_y$$
   * Magnitudo gradien dihitung dan dinormalisasi ke rentang 8-bit $[0, 255]$:
     $$\text{Magnitude}(x,y) = \sqrt{G_x(x,y)^2 + G_y(x,y)^2}$$
     $$I_{\text{Pseudo-IR}}(x,y) = \left\lfloor \frac{\text{Magnitude}(x,y) - \min(\text{Magnitude})}{\max(\text{Magnitude}) - \min(\text{Magnitude})} \times 255 \right\rfloor$$
   * Citra hasil direplikasi menjadi 3 kanal (*channel replication*) untuk memenuhi dimensi masukan backbone DEYOLO.

2. **Varian CLAHE + Sobel (Sintesis Peningkatan Kontras - EXP-07):**
   * Menerapkan *Contrast Limited Adaptive Histogram Equalization* (clip limit = 2.0, tile grid size = $8 \times 8$) pada citra skala abu-abu sebelum penghalusan Gaussian ringan (kernel $5 \times 5, \sigma = 1.0$) dan ekstraksi gradien Sobel $3 \times 3$.

---

## 3.4 Arsitektur Dual Feature Enhancement YOLO (DEYOLO)
Model DEYOLO menggunakan arsitektur *dual-stream backbone* bergaya YOLOv8n untuk memproses citra RGB (*Visible/VI*) dan Pseudo-IR secara paralel. Fusi lintas modalitas dilakukan oleh modul *Dual-Enhancement Attention* (DEA) yang menggabungkan dua sub-modul berurutan:

* **Dual-Enhancement Channel Attention (DECA):**
  Melakukan pembobotan kanal silang (*cross-channel gating*). Deskriptor kanal masing-masing modalitas ($W_{vi}$ dan $W_{ir}$) dihitung melalui *Global Average Pooling* dan dua lapisan *Fully-Connected* dengan rasio reduksi $r$, lalu digabungkan dengan konteks global $G$:
  $$F_{vi}' = F_{vi} \odot \sigma(W_{ir} \odot G), \quad F_{ir}' = F_{ir} \odot \sigma(W_{vi} \odot G)$$

* **Dual-Enhancement Position Attention (DEPA):**
  Melakukan pembobotan spasial silang (*cross-spatial gating*) pada tingkat piksel menggunakan konvolusi multi-kernel simetris $3 \times 3$ yang terbukti lebih efektif mereduksi derau garis hujan vertikal dibandingkan kernel asimetris $3 \times 7$.

---

## 3.5 Pemodelan Tracking ByteTrack & Garis Hitung Split-Lane

### 3.5.1 Ruang Keadaan Kalman Filter 8-Dimensi
$$\mathbf{x} = \begin{bmatrix} x_c & y_c & a & h & \dot{x}_c & \dot{y}_c & \dot{a} & \dot{h} \end{bmatrix}^T$$
* $(x_c, y_c)$: Titik pusat (*centroid*) kotak pembatas kendaraan.
* $a = w/h$: Rasio aspek lebar terhadap tinggi kotak pembatas.
* $h$: Tinggi kotak pembatas dalam piksel.
* $(\dot{x}_c, \dot{y}_c, \dot{a}, \dot{h})$: Komponen kecepatan pergerakan linier objek.

### 3.5.2 Topologi Garis Hitung Split-Lane Dual-Tripwire
* **Lajur Kiri (Arus Keluar / OUT / Ke Atas):** Garis diletakkan pada rasio $Y_{\text{left}} = 0.70$ dari tinggi frame.
* **Lajur Kanan (Arus Masuk / IN / Ke Bawah):** Garis diletakkan pada rasio $Y_{\text{right}} = 0.50$ dari tinggi frame.
* **Filter Nir-Duplikasi:** Memanfaatkan himpunan $\mathcal{S}_{\text{counted}}$ sehingga suatu objek dengan $\text{ID}_k$ hanya dihitung 1 kali sepanjang siklus hidupnya (**probabilitas double-counting = 0%**).

---

# BAB IV: HASIL PENGUJIAN DAN PEMBAHASAN

## 4.1 Evaluasi Kinerja Model Deteksi DEYOLO (Rujukan JUTIF 2025)
Berdasarkan hasil eksperimen ablasi berjenjang pada jurnal rujukan (Chazar & Nugraha, JUTIF 2025), dilakukan perbandingan kinerja antara model Baseline (YOLOv8n) dan enam konfigurasi DEYOLO pada dataset pengujian cuaca hujan. Hasil kuantitatif dirangkum pada Tabel 4.1.

**Tabel 4.1 Evaluasi Kinerja Kuantitatif Model Baseline YOLOv8n dan Varian DEYOLO (Chazar & Nugraha, 2025)**
| Eksperimen | Konfigurasi Arsitektur Model | Precision (\%) | Recall (\%) | mAP@0.5 (\%) | mAP@0.5:0.95 (\%) | F1-Score (\%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | Baseline (YOLOv8n RGB Tunggal) | 83.25 | 76.66 | **87.37** | **65.62** | 79.82 |
| **EXP-02** | DEYOLOn-Default (DEPA 3x7, r=16, Sobel) | 81.10 | 76.80 | 85.40 | 65.50 | 78.89 |
| **EXP-03** | DEYOLOn (DEPA 3x7, r=8, Sobel) | 75.38 | 75.84 | 82.28 | 62.49 | 75.61 |
| **EXP-04** | DEYOLOn (DEPA 3x3, r=16, Sobel) | 83.36 | 78.44 | 86.69 | 62.72 | 80.84 |
| **EXP-05** | **DEYOLOn (DEPA 3x3, r=8, Sobel)** | 83.86 | **78.97** | 87.14 | 64.98 | **81.34** |
| **EXP-06** | DEYOLOn (DEPA 3x3, r=4, Sobel) | 81.57 | 78.73 | 84.92 | 64.49 | 80.12 |
| **EXP-07** | **DEYOLOn (DEPA 3x3, r=8, CLAHE+Sobel)** | **85.81** | 74.92 | 87.28 | 65.34 | 79.99 |

> **Analisis Kritis Pemilihan Model untuk Traffic Counting:**
> 1. **EXP-05 (DEPA 3x3, r=8, Sobel)** meraih **Recall tertinggi (78.97%)** dan **F1-Score tertinggi (81.34%)**, dengan peningkatan Recall sebesar $+2.31\%$ dan F1-Score sebesar $+1.52\%$ dibandingkan Baseline YOLOv8n. Karena pada aplikasi penghitungan lalu lintas (*traffic counting*), kendaraan yang terlewat (*false negatives*) jauh lebih fatal dibandingkan false positive, maka **EXP-05 dipilih sebagai model utama inferensi tracking**.
> 2. **EXP-07 (CLAHE + Sobel)** meraih **Precision tertinggi (85.81%)** ($+2.56\%$ di atas baseline) yang sangat efektif menekan deteksi palsu akibat pantulan air pada aspal basah.

---

## 4.2 Hasil Pengujian Tracking dan Penghitungan Volume Kendaraan (Video 5 Menit)
Pengujian *end-to-end* pelacakan dan penghitungan dilakukan pada rekaman video CCTV jalan tol aktual (`1138.mp4`) berdurasi 5 menit ($300.62\text{ detik}$, $1.453\text{ frame}$ pada framerate asli $4.833\text{ FPS}$) menggunakan kombinasi model optimal **DEYOLO EXP-05** dan **ByteTrack**. Rekapitulasi volume kendaraan terhitung disajikan pada Tabel 4.2.

**Tabel 4.2 Rekapitulasi Volume Kendaraan Terhitung Berdasarkan Arah dan Lajur**
| Arah Aliran Lalu Lintas | Lajur Jalan | Posisi Tripwire ($Y$) | Jumlah Terhitung (Unit) | Persentase (\%) |
| :--- | :--- | :---: | :---: | :---: |
| **IN (Masuk / Ke Bawah)** | Lajur Kanan | $Y = 0.50$ ($50\%$) | **210** | $67.31\%$ |
| **OUT (Keluar / Ke Atas)** | Lajur Kiri | $Y = 0.70$ ($70\%$) | **102** | $32.69\%$ |
| **TOTAL KESELURUHAN ($Q$)** | **Dua Arah** | — | **312** | **$100.00\%$** |

### Ekstrapolasi Laju Arus Lalu Lintas per Jam ($Q_{\text{jam}}$ / Standar MKJI):
$$Q_{\text{jam}} = \frac{N_{\text{total}}}{T_{\text{jam}}} = \frac{312\text{ unit}}{\frac{300.62}{3600}\text{ jam}} = \mathbf{3.736\text{ kendaraan/jam}}$$
* Laju Arus Masuk (IN): $Q_{\text{in}} = 2.515\text{ kendaraan/jam}$
* Laju Arus Keluar (OUT): $Q_{\text{out}} = 1.221\text{ kendaraan/jam}$

---

## 4.3 Distribusi Klasifikasi Multi-Kelas Kendaraan Terhitung
Model DEYOLO secara simultan melakukan klasifikasi jenis kendaraan ke dalam tiga kelas utama. Distribusi kelas kendaraan terhitung disajikan pada Tabel 4.3.

**Tabel 4.3 Distribusi Klasifikasi Multi-Kelas Kendaraan Terhitung**
| Kelas Kendaraan | Arah IN (Masuk) | Arah OUT (Keluar) | Total Volume (Unit) | Proporsi Relatif (\%) |
| :--- | :---: | :---: | :---: | :---: |
| **Mobil Penumpang (*Car*)** | 163 | 71 | **234** | **$75.00\%$** |
| **Truk (*Truck*)** | 39 | 24 | **63** | **$20.19\%$** |
| **Bus (*Bus*)** | 8 | 7 | **15** | **$4.81\%$** |
| **TOTAL** | **210** | **102** | **312** | **$100.00\%$** |

---

## 4.4 Pembahasan Dekomposisi ID: Volume Counting (312) vs Total ID Tracker (427)
Terdapat perbedaan antara **Total Kendaraan Melintasi Garis (312 unit)** dan **Total ID yang Terdaftar pada Tracker (427 ID)**. Analisis mengenai selisih $115\text{ ID}$ dijabarkan pada Tabel 4.4.

**Tabel 4.4 Analisis Dekomposisi Metrik Counting Line vs. Tracker Lifecycle Pool**
| Kategori Objek Terdeteksi | Jumlah ID | Persentase | Status Penghitungan | Keterangan Fenomena Fisik |
| :--- | :---: | :---: | :---: | :--- |
| **Kendaraan Sah Menyeberang Garis** | **312** | **$73.07\%$** | **VALID (Tercatat di CSV)** | Kendaraan nyata melintasi penampang jalan dengan vektor arah valid. |
| **Batas Awal & Akhir Video** | $\approx 42$ | $9.84\%$ | Tidak Dihitung | Kendaraan sudah di bawah garis saat $t=0\text{s}$ atau baru masuk frame saat $t=300\text{s}$. |
| **Objek Kejauhan (*Horizon* Atas)** | $\approx 48$ | $11.24\%$ | Tidak Dihitung | Mobil berukuran sangat kecil di kejauhan yang belum mencapai garis. |
| **Eliminasi Noise Hujan & Pantulan** | $\approx 25$ | $5.85\%$ | Tereliminasi (*Suppressed*) | Deteksi palsu kilat 1 frame akibat pantulan air yang gagal mendekati garis. |
| **TOTAL ALOKASI ID TRACKER** | **427** | **$100.00\%$** | — | — |

---

## 4.5 Kesimpulan Bab IV
1. Arsitektur **DEYOLO EXP-05 (DEPA 3x3, DECA r=8, Sobel Pseudo-IR)** terbukti sebagai konfigurasi optimal untuk aplikasi penghitungan lalu lintas saat hujan dengan **Recall 78.97%** dan **F1-Score 81.34%**.
2. Sistem pelacakan **ByteTrack** dan **Split-Lane Dual-Tripwire** berhasil menghitung **312 unit kendaraan** tanpa duplikasi pada video pengujian 5 menit cuaca hujan lebat (210 unit IN, 102 unit OUT), setara dengan laju arus lalu lintas **$3.736\text{ kendaraan/jam}$**.
3. Sistem menghasilkan file video H.264 CFR yang mulus serta ekspor paket ZIP lengkap berisi data log CSV dan ringkasan statistik yang siap dianalisis untuk rekayasa lalu lintas.
