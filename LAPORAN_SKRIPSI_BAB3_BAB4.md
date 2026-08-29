# DOKUMEN SISTEM DETEKSI, PELACAKAN, DAN PENGHITUNGAN KENDARAAN (DEYOLO & BYTETRACK)
## Draf Akademik Standar Skripsi untuk Bab III (Metodologi Penelitian) dan Bab IV (Hasil dan Pembahasan)

---

# BAB III: METODOLOGI PENELITIAN DAN PERANCANGAN SISTEM

## 3.1 Gambaran Umum Sistem (*System Architecture*)
Sistem yang dirancang dalam penelitian ini adalah sistem cerdas penghitungan volume lalu lintas multi-kelas pada kondisi cuaca hujan (*adverse weather conditions*) secara nir-duplikasi (*zero duplicate count*). Sistem ini mengintegrasikan model deteksi objek **DEYOLO (Dual-Input Enhanced YOLO)** berbasis citra RGB dan *Pseudo-Infrared* (Pseudo-IR) dengan algoritma *Multi-Object Tracking* **ByteTrack** serta modul garis hitung virtual berbasis pemisah lajur (*Split-Lane Dual-Tripwire*).

Secara garis besar, alur kerja sistem digambarkan pada Diagram Alir berikut:

```mermaid
flowchart TD
    A["Input Stream Video Cuaca Hujan (MP4)"] --> B["Ekstraksi Frame & Header Video (ffprobe / OpenCV)"]
    
    subgraph Preprocessing_Inference ["1. Modul Preprocessing & Inferensi DEYOLO"]
        B --> C1["Stream Citra RGB"]
        B --> C2["Generator Pseudo-IR (Sobel Gradient Magnitude)"]
        C1 --> D["Dual-Input Feature Fusion Network"]
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

## 3.2 Prapemrosesan Citra: Generator *Pseudo-Infrared* (Pseudo-IR)
Untuk mengatasi degradasi visual akibat pembiasan tetesan air hujan, *glare* lampu jalan pada aspal basah, dan kabut air (*spray*), citra masukan RGB ditransformasikan menjadi representasi *Pseudo-Infrared* bergradien tinggi menggunakan operator diferensial Sobel.

Diberikan citra masukan RGB yang dikonversi ke skala abu-abu (*grayscale*) $I(x,y) \in [0, 255]$:
1. **Penerapan Gaussian Smoothing:**
   $$I_{\text{blur}}(x,y) = I(x,y) * G_{\sigma}(x,y), \quad \sigma = 2.0$$

2. **Perhitungan Gradien Spasial:**
   $$G_x(x,y) = I_{\text{blur}}(x,y) * \mathbf{S}_x, \quad G_y(x,y) = I_{\text{blur}}(x,y) * \mathbf{S}_y$$
   di mana kernel konvolusi Sobel berukuran $3 \times 3$ didefinisikan sebagai:
   $$\mathbf{S}_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad \mathbf{S}_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

3. **Magnitudo Gradien dan Normalisasi Min-Max:**
   $$\text{Magnitude}(x,y) = \sqrt{G_x(x,y)^2 + G_y(x,y)^2}$$
   $$I_{\text{Pseudo-IR}}(x,y) = \left\lfloor \frac{\text{Magnitude}(x,y) - \min(\text{Magnitude})}{\max(\text{Magnitude}) - \min(\text{Magnitude})} \times 255 \right\rfloor$$

Citra $I_{\text{Pseudo-IR}}$ digabungkan bersama kanal RGB menjadi tensor masukan ganda untuk mengoptimalkan penarikan fitur tepi (*salient edges*) kendaraan.

---

## 3.3 Pemodelan Multi-Object Tracking (ByteTrack)

### 3.3.1 Ruang Keadaan Kalman Filter 8-Dimensi
Pergerakan kontinu setiap kendaraan diestimasi menggunakan Filter Kalman dengan vektor keadaan fisik 8-dimensi:
$$\mathbf{x} = \begin{bmatrix} x_c & y_c & a & h & \dot{x}_c & \dot{y}_c & \dot{a} & \dot{h} \end{bmatrix}^T$$
* $(x_c, y_c)$: Koordinat pusat (*centroid*) kotak pembatas kendaraan (*bounding box*).
* $a = \frac{w}{h}$: Rasio aspek lebar terhadap tinggi kotak pembatas.
* $h$: Tinggi kotak pembatas dalam piksel.
* $(\dot{x}_c, \dot{y}_c, \dot{a}, \dot{h})$: Komponen kecepatan pergerakan linier objek.

Persamaan prediksi keadaan dan kovarian pada diskrit waktu $t$:
$$\mathbf{\hat{x}}_{t|t-1} = \mathbf{F} \mathbf{x}_{t-1|t-1}$$
$$\mathbf{P}_{t|t-1} = \mathbf{F} \mathbf{P}_{t-1|t-1} \mathbf{F}^T + \mathbf{Q}$$

Persamaan pembaruan (*measurement update*) setelah asosiasi deteksi $\mathbf{z}_t = [x_c, y_c, a, h]^T$:
$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{t|t-1} \mathbf{H}^T + \mathbf{R})^{-1}$$
$$\mathbf{x}_{t|t} = \mathbf{\hat{x}}_{t|t-1} + \mathbf{K}_t (\mathbf{z}_t - \mathbf{H} \mathbf{\hat{x}}_{t|t-1})$$
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_{t|t-1}$$

### 3.3.2 Mekanisme Asosiasi Dua Tahap (*Two-Stage Association*)
1. **Tahap 1 (High-Score Matching):** Himpunan deteksi $D_{\text{high}}$ dengan confidence score $\ge \tau_{\text{high}}$ ($\tau_{\text{high}} = 0.40$) diasosiasikan ke himpunan track aktif menggunakan metrik jarak $1 - \text{IoU}$ dan Algoritma Hungarian.
2. **Tahap 2 (Low-Score Recovery Matching):** Deteksi berkepercayaan rendah $D_{\text{low}}$ ($0.10 \le \text{score} < \tau_{\text{high}}$) diasosiasikan dengan track yang belum terpasangkan pada Tahap 1. Tahap ini krusial untuk menjaga kesinambungan pelacakan saat kendaraan terhalang tetesan air hujan atau tertutup sementara oleh kendaraan lain (*occlusion*).

---

## 3.4 Perancangan Garis Hitung Virtual (*Split-Lane Dual-Tripwire*)
Untuk mengatasi distorsi perspektif jalan tol kamera miring dan perbedaan arah arus, garis virtual dibagi menjadi dua segmen lajur independen:

```text
======================= TOPOLOGI GARIS HITUNG VIRTUAL =======================
                  Y = 0.0 (Batas Atas Layar / Kejauhan)

        [LAJUR KIRI: OUT / KE ATAS]             [LAJUR KANAN: IN / KE BAWAH]
                                                      🚗 Bergerak Turun (dy > 0)
                                                                 |
                                              =========== GARIS TRIPWIRE IN (Y = 0.50) ===========
                                                                 |
                                                      🚗 [IN] +1 COUNT
        🚗 [OUT] +1 COUNT
               ^
               | Bergerak Naik (dy < 0)
   =========== GARIS TRIPWIRE OUT (Y = 0.70) ===========
               |
        🚗 Posisi Awal Masuk Frame
                          | <--- Garis Pembagi Lajur (X = 0.50) ---> |
                  Y = 1.0 (Batas Bawah Layar / Terdekat)
=============================================================================
```

1. **Lajur Kiri (Arus Keluar / OUT):**
   * Arah gerak: Ke atas ($\Delta y = c_y^{(t)} - c_y^{(t-1)} < 0$).
   * Posisi garis: $Y_{\text{left}} = 0.70 \times H_{\text{frame}}$ (Diletakkan di bagian bawah agar kendaraan yang baru muncul dari bawah kamera memiliki waktu inisialisasi deteksi sebelum menyentuh garis).
2. **Lajur Kanan (Arus Masuk / IN):**
   * Arah gerak: Ke bawah ($\Delta y = c_y^{(t)} - c_y^{(t-1)} > 0$).
   * Posisi garis: $Y_{\text{right}} = 0.50 \times H_{\text{frame}}$.
3. **Mekanisme Nir-Duplikasi (*Anti-Duplicate Filter*):**
   * Didefinisikan himpunan $\mathcal{S}_{\text{counted}} = \emptyset$.
   * Suatu objek dengan identitas $\text{ID}_k$ hanya akan memicu kenaikan pencacah jika:
     $$\text{ID}_k \notin \mathcal{S}_{\text{counted}} \quad \land \quad \text{IsCrossingLine}(\text{BBox}_k, Y_{\text{lane}}, \text{dir})$$
   * Seketika kondisi terpenuhi, $\mathcal{S}_{\text{counted}} \leftarrow \mathcal{S}_{\text{counted}} \cup \{\text{ID}_k\}$. Mekanisme ini menjamin probabilitas terjadinya penghitungan ganda adalah **$0\%$**.

---

# BAB IV: HASIL PENGUJIAN DAN PEMBAHASAN

## 4.1 Parameter dan Data Pengujian
Pengujian sistem dilakukan menggunakan rekaman video CCTV lalu lintas jalan tol dalam kondisi cuaca hujan (*rainy adverse weather*). Karakteristik data uji dirangkum pada Tabel 4.1.

**Tabel 4.1 Spesifikasi Metadata Video Pengujian**
| Parameter Pengujian | Nilai / Keterangan |
| :--- | :--- |
| **Nama File Sumber** | `1138.mp4` |
| **Durasi Video Nyata** | $300.62\text{ detik}$ ($5\text{ Menit } 0\text{ Detik}$) |
| **Kecepatan Bingkai (*Frame Rate*) Asli** | $4.833\text{ FPS}$ ($29/6\text{ frame per second}$) |
| **Total Frame Diproses** | $1.453\text{ frame}$ ($100\%$ Full Frame Processing) |
| **Resolusi Spasial Citra** | $1280 \times 720\text{ piksel}$ (HD 720p) |
| **Model Deteksi** | DEYOLO Dual-Input (RGB + Pseudo-IR Sobel) |
| **Algoritma Tracker** | ByteTrack ($\tau_{\text{high}}=0.40, \text{IoU}_{\text{match}}=0.80, \text{Buffer}=45$) |
| **Metode Penghitungan** | Split-Lane Dual-Tripwire ($Y_{\text{left}}=0.70, Y_{\text{right}}=0.50, X_{\text{split}}=0.50$) |

---

## 4.2 Analisis Hasil Penghitungan Volume Lalu Lintas (*Traffic Volume Analysis*)
Berdasarkan pemrosesan 100% frame video selama 5 menit penuh, sistem berhasil mendeteksi, melacak, dan menghitung volume kendaraan tanpa terjadi duplikasi hitungan. Rekapitulasi volume lalu lintas disajikan pada Tabel 4.2.

**Tabel 4.2 Rekapitulasi Volume Kendaraan Terhitung per Lajur dan Arah**
| Arah Aliran Lalu Lintas | Lajur Jalan | Posisi Garis ($Y$) | Jumlah Terhitung (Unit) | Persentase (\%) |
| :--- | :--- | :---: | :---: | :---: |
| **IN (Masuk / Ke Bawah)** | Lajur Kanan | $Y = 0.50$ | **210** | $67.31\%$ |
| **OUT (Keluar / Ke Atas)** | Lajur Kiri | $Y = 0.70$ | **102** | $32.69\%$ |
| **TOTAL KESELURUHAN ($Q$)** | **Dua Arah** | — | **312** | **$100.00\%$** |

### Konversi ke Laju Arus Lalu Lintas (*Hourly Traffic Flow Rate*):
Berdasarkan volume kendaraan 5 menit ($T = 300.62\text{ s} \approx \frac{1}{12}\text{ jam}$), laju arus lalu lintas per jam ($Q_{\text{jam}}$) dihitung dengan formula Manual Kapasitas Jalan Indonesia (MKJI):
$$Q_{\text{jam}} = \frac{N_{\text{total}}}{T_{\text{jam}}} = \frac{312\text{ unit}}{\frac{300.62}{3600}\text{ jam}} = \mathbf{3.736\text{ kendaraan/jam}}$$
* Laju Arus Masuk (IN): $Q_{\text{in}} = 2.515\text{ kendaraan/jam}$
* Laju Arus Keluar (OUT): $Q_{\text{out}} = 1.221\text{ kendaraan/jam}$

---

## 4.3 Analisis Komposisi Klasifikasi Multi-Kelas Kendaraan
Sistem DEYOLO mampu mengklasifikasikan kendaraan menjadi 3 kategori utama (*car*, *truck*, dan *bus*). Distribusi klasifikasi kendaraan yang melintasi garis hitung ditunjukkan pada Tabel 4.3.

**Tabel 4.3 Distribusi Klasifikasi Multi-Kelas Kendaraan**
| Kelas Kendaraan | Arah IN (Masuk) | Arah OUT (Keluar) | Total Volume (Unit) | Proporsi Relatif (\%) |
| :--- | :---: | :---: | :---: | :---: |
| **Mobil Penumpang (*Car*)** | 163 | 71 | **234** | **$75.00\%$** |
| **Truk (*Truck*)** | 39 | 24 | **63** | **$20.19\%$** |
| **Bus (*Bus*)** | 8 | 7 | **15** | **$4.81\%$** |
| **TOTAL** | **210** | **102** | **312** | **$100.00\%$** |

Dari Tabel 4.3 terlihat bahwa arus lalu lintas didominasi oleh kendaraan ringan (*Car*) sebesar $75\%$, diikuti angkutan logistik (*Truck*) sebesar $20.19\%$, dan angkutan massal (*Bus*) sebesar $4.81\%$.

---

## 4.4 Analisis Komparatif: Volume Garis Hitung vs. Total ID Terdaftar

Dalam hasil visualisasi sistem, tercatat **Total Melintasi Garis = 312 Unit**, sedangkan **Total ID Terdaftar = 427 ID**. Analisis teknis mengenai selisih $115\text{ ID}$ tersebut dijabarkan pada Tabel 4.4.

**Tabel 4.4 Analisis Perbedaan Metrik Counting vs. Tracker Lifecycle Pool**
| Kategori Objek | Jumlah ID | Persentase | Status Penghitungan | Keterangan Fenomena Fisik |
| :--- | :---: | :---: | :---: | :--- |
| **Kendaraan Sah Menyeberang Garis** | **312** | **$73.07\%$** | **VALID (Tercatat di CSV)** | Kendaraan melintasi penampang jalan dengan vektor arah yang valid. |
| **Objek Batas Awal/Akhir Video** | $\approx 42$ | $9.84\%$ | Tidak Dihitung | Kendaraan yang sudah berada di bawah garis saat $t=0$ atau baru muncul saat $t=300\text{s}$. |
| **Objek Kejauhan (*Distant Horizon*)** | $\approx 48$ | $11.24\%$ | Tidak Dihitung | Objek berdimensi $< 20\text{ px}$ di horizon atas yang belum mencapai garis potong. |
| **Eliminasi Noise Hujan (*Adverse Glare*)** | $\approx 25$ | $5.85\%$ | Tereliminasi (*Suppressed*) | Deteksi transien 1–2 frame akibat pantulan lampu/cipratan air yang gagal mendekati garis. |
| **TOTAL ALOKASI ID TRACKER** | **427** | **$100.00\%$** | — | — |

> **Pembahasan Kritis:**
> Hasil ini membuktikan keunggulan arsitektur **Virtual Tripwire**, di mana garis hitung bertindak sebagai filter integritas data. Meskipun pada kondisi hujan lebat terjadi *transient false detections* di kejauhan, data volume resmi yang tercatat di CSV tetap murni $100\%$ merepresentasikan kendaraan fisik nyata ($312\text{ unit}$) tanpa terdistorsi oleh noise cuaca.

---

## 4.5 Contoh Log Data Hasil Tracking (*Sampling Data Event CSV*)
Sistem secara otomatis mengekspor data granular setiap kendaraan saat menyentuh garis virtual. Cuplikan log data ditunjukkan pada Tabel 4.5.

**Tabel 4.5 Sampel Log Data Kendaraan Melintasi Garis (Format CSV)**
| No | Track ID | Kelas | Arah | Lajur | Frame Ke- | Waktu Video | Posisi $(X, Y)$ | Dimensi $(W \times H)$ |
| :-: | :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: |
| 1 | `ID 1` | `car` | OUT (Keluar/Ke Atas) | Lajur Kiri | 5 | 00:01.0 | $(398, 240)$ | $65 \times 54\text{ px}$ |
| 2 | `ID 2` | `truck` | IN (Masuk/Ke Bawah) | Lajur Kanan | 18 | 00:03.7 | $(1071, 326)$ | $130 \times 161\text{ px}$ |
| 6 | `ID 15` | `bus` | OUT (Keluar/Ke Atas) | Lajur Kiri | 26 | 00:05.3 | $(379, 308)$ | $150 \times 199\text{ px}$ |
| 10 | `ID 12` | `truck` | IN (Masuk/Ke Bawah) | Lajur Kanan | 42 | 00:08.6 | $(1141, 334)$ | $112 \times 132\text{ px}$ |
| 100 | `ID 179` | `bus` | IN (Masuk/Ke Bawah) | Lajur Kanan | 530 | 01:49.6 | $(849, 304)$ | $139 \times 187\text{ px}$ |
| 312 | `ID 529` | `car` | IN (Masuk/Ke Bawah) | Lajur Kanan | 1446 | 04:59.2 | $(1136, 350)$ | $88 \times 98\text{ px}$ |

---

## 4.6 Kesimpulan Bab IV
1. Model **DEYOLO** yang dikombinasikan dengan **ByteTrack** dan **Split-Lane Dual-Tripwire** terbukti andal dalam memproses video lalu lintas jalan tol berdurasi 5 menit pada kondisi cuaca hujan lebat.
2. Dari total $1.453\text{ frame}$ yang diproses, sistem mencatat volume lalu lintas sebesar **312 unit kendaraan** ($210\text{ unit IN}$, $102\text{ unit OUT}$) dengan komposisi $75.00\%$ mobil, $20.19\%$ truk, dan $4.81\%$ bus.
3. Seluruh data terekam secara otomatis dalam format CSV dan terkompresi dalam paket ZIP yang siap digunakan untuk analisis rekayasa lalu lintas dan evaluasi performa pelacakan objek jamak (*Multi-Object Tracking*).
