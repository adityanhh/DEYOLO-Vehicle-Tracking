import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_thesis_docx(output_path):
    doc = docx.Document()
    
    # Standar Margin Skripsi (Top 3cm, Bottom 3cm, Left 4cm, Right 3cm)
    for section in doc.sections:
        section.top_margin = Inches(1.18)    # 3 cm
        section.bottom_margin = Inches(1.18) # 3 cm
        section.left_margin = Inches(1.57)   # 4 cm
        section.right_margin = Inches(1.18)  # 3 cm

    # Base Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.5
    normal_style.paragraph_format.space_after = Pt(6)

    # Document Header / Judul
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t1 = p_title.add_run("DOKUMEN METODOLOGI DAN EVALUASI HASIL PENELITIAN SKRIPSI\n")
    run_t1.bold = True
    run_t1.font.size = Pt(14)
    run_t2 = p_title.add_run(
        "DETEKSI, PELACAKAN, DAN PENGHITUNGAN KENDARAAN PADA KONDISI HUJAN\n"
        "MENGGUNAKAN DUAL FEATURE ENHANCEMENT YOLO (DEYOLO) DAN BYTETRACK\n"
    )
    run_t2.bold = True
    run_t2.font.size = Pt(13)
    run_t3 = p_title.add_run(
        "Rujukan Ilmiah: Jurnal Teknik Informatika (JUTIF) Vol. 6, No. 6, Desember 2025\n"
        "Penulis: Chalifa Chazar, Aditya Nugraha (Institut Teknologi Nasional Bandung)\n"
        "DOI: https://doi.org/10.52436/1.jutif.6.6.3540"
    )
    run_t3.italic = True
    run_t3.font.size = Pt(10.5)
    p_title.paragraph_format.space_after = Pt(20)

    # =========================================================================
    # BAB III
    # =========================================================================
    p_b3 = doc.add_paragraph()
    p_b3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_b3 = p_b3.add_run("BAB III\nMETODOLOGI PENELITIAN DAN PERANCANGAN SISTEM")
    r_b3.bold = True
    r_b3.font.size = Pt(13)
    p_b3.paragraph_format.space_after = Pt(14)

    # 3.1
    p = doc.add_paragraph()
    r = p.add_run("3.1 Gambaran Umum Sistem (System Architecture)")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Sistem yang dirancang dalam penelitian ini adalah sistem cerdas penghitungan volume lalu lintas multi-kelas (car, bus, truck) pada kondisi cuaca buruk (adverse weather), khususnya kondisi hujan lebat di jalan tol. Sistem ini mengintegrasikan model deteksi objek DEYOLO (Dual-Feature Enhancement YOLO) berbasis citra RGB dan Pseudo-Infrared (Pseudo-IR) dengan algoritma Multi-Object Tracking (MOT) ByteTrack serta modul garis hitung virtual berbasis pemisah lajur (Split-Lane Dual-Tripwire)."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p = doc.add_paragraph(
        "Arsitektur alur kerja sistem terdiri atas empat tahapan utama: (1) Prapemrosesan citra sintetis Pseudo-IR (Sobel Edge dan CLAHE), (2) Ekstraksi fitur dan fusi atensi ganda DECA (Dual-Enhancement Channel Attention) serta DEPA (Dual-Enhancement Position Attention) pada DEYOLO, (3) Estimasi pergerakan spasial dan asosiasi dua tahap menggunakan ByteTrack dengan Kalman Filter 8-dimensi, serta (4) Evaluasi penyeberangan garis virtual Split-Lane Tripwire yang dilengkapi filter anti-duplikasi dan pencatatan log data otomatis ke format CSV."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # 3.2
    p = doc.add_paragraph()
    r = p.add_run("3.2 Dataset dan Anotasi (Dataset and Annotation)")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Dataset penelitian dibangun dari rekaman video CCTV jalan tol pada kondisi cuaca hujan siang hari. Tiga klip video mentah, masing-masing berdurasi sekitar 5 menit, diekstraksi dengan laju pengambilan sampel tetap 3 frame per detik (3 FPS). Total frame yang dianotasi adalah sebanyak 2.764 frame yang dibagi ke dalam tiga subset (Train, Validation, Testing) secara independen untuk memastikan evaluasi objektif pada kondisi rekaman yang belum pernah dilihat model (unseen conditions). Distribusi dataset disajikan pada Tabel 3.1."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Table 3.1
    p_tbl_ds = doc.add_paragraph()
    r_tbl_ds = p_tbl_ds.add_run("Tabel 3.1 Distribusi Citra dan Jumlah Instance Objek per Kelas pada Dataset")
    r_tbl_ds.bold = True
    p_tbl_ds.paragraph_format.space_after = Pt(4)

    tbl_ds = doc.add_table(rows=5, cols=6)
    tbl_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_ds_data = [
        ("Subset", "Jumlah Gambar", "Mobil (Car)", "Bus (Bus)", "Truk (Truck)", "Total Instance"),
        ("Train (Pelatihan)", "2.389", "7.858", "451", "2.614", "10.923"),
        ("Validation (Validasi)", "316", "744", "72", "270", "1.086"),
        ("Testing (Pengujian)", "59", "147", "18", "48", "213"),
        ("TOTAL DATASET", "2.764", "8.749", "541", "2.932", "12.222")
    ]
    for i, row in enumerate(tbl_ds.rows):
        for j, val in enumerate(tbl_ds_data[i]):
            row.cells[j].text = val
            set_cell_margins(row.cells[j], top=70, bottom=70, left=90, right=90)
            if i == 0 or i == 4:
                set_cell_background(row.cells[j], 'E8EEF5' if i == 0 else 'F2F4F7')
                row.cells[j].paragraphs[0].runs[0].bold = True

    p = doc.add_paragraph(
        "Seluruh anotasi kotak pembatas (bounding box) dihasilkan menggunakan Computer Vision Annotation Tool (CVAT) dengan metode interpolasi antar-frame, kemudian diekspor ke format standar label YOLO."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # 3.3
    p = doc.add_paragraph()
    r = p.add_run("3.3 Prapemrosesan Citra: Sintesis Modalitas Pseudo-Infrared (Pseudo-IR)")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Kondisi hujan lebat menimbulkan gangguan visual berupa garis hujan vertikal (rain streaks), cipratan air roda kendaraan (spray), dan pantulan lampu pada aspal basah (specular reflections). Untuk mengatasi ketiadaan sensor inframerah termal fisik, modalitas sintetis Pseudo-IR dibentuk melalui dua metode prapemrosesan:"
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p = doc.add_paragraph(
        "1. Varian Sobel Edge (Sintesis Utama - EXP-05):\n"
        "   Citra RGB dikonversi ke skala abu-abu I_gray, dihaluskan dengan Gaussian Blur (sigma = 2.0) untuk mereduksi derau frekuensi tinggi, lalu diekstraksi gradien tepi horizontal dan vertikal menggunakan kernel Sobel 3x3:\n"
        "   G_x = I_blur * S_x,    G_y = I_blur * S_y\n"
        "   Magnitude(x, y) = sqrt(G_x^2 + G_y^2)\n"
        "   Hasil magnitudo dinormalisasi ke rentang 8-bit [0, 255] dan direplikasi menjadi 3 kanal identik.\n\n"
        "2. Varian CLAHE + Sobel (Sintesis Peningkatan Kontras - EXP-07):\n"
        "   Menerapkan Contrast Limited Adaptive Histogram Equalization (clip limit = 2.0, tile grid = 8x8) pada citra skala abu-abu sebelum penghalusan Gaussian ringan (kernel 5x5, sigma = 1.0) dan ekstraksi gradien Sobel 3x3."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)

    # 3.4
    p = doc.add_paragraph()
    r = p.add_run("3.4 Arsitektur Dual Feature Enhancement YOLO (DEYOLO)")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Model DEYOLO menggunakan arsitektur backbone ganda (dual-stream backbone) bergaya YOLOv8n untuk memproses citra RGB (Visible/VI) dan Pseudo-IR secara paralel. Fusi lintas modalitas dilakukan oleh modul Dual-Enhancement Attention (DEA) yang menggabungkan dua sub-modul berurutan:"
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p = doc.add_paragraph(
        "• Dual-Enhancement Channel Attention (DECA):\n"
        "  Melakukan pembobotan kanal silang (cross-channel gating). Deskriptor kanal masing-masing modalitas (W_vi dan W_ir) dihitung melalui Global Average Pooling dan dua lapisan Fully-Connected dengan rasio reduksi r, lalu digabungkan dengan konteks global G:\n"
        "  F_vi' = F_vi ⊙ sigma(W_ir ⊙ G)\n"
        "  F_ir' = F_ir ⊙ sigma(W_vi ⊙ G)\n\n"
        "• Dual-Enhancement Position Attention (DEPA):\n"
        "  Melakukan pembobotan spasial silang (cross-spatial gating) pada tingkat piksel menggunakan konvolusi multi-kernel simetris 3x3 yang terbukti lebih efektif mereduksi derau garis hujan vertikal dibandingkan kernel asimetris 3x7."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)

    # 3.5
    p = doc.add_paragraph()
    r = p.add_run("3.5 Multi-Object Tracking (ByteTrack) dan Split-Lane Tripwire")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Untuk pelacakan objek jamak, ByteTrack memodelkan dinamika fisik pergerakan kendaraan menggunakan Kalman Filter 8-keadaan [x_c, y_c, a, h, v_xc, v_yc, v_a, v_h]^T dan mekanisme asosiasi dua tahap (two-stage association). Kotak deteksi berkepercayaan rendah (0.10 <= conf < 0.40) tetap dipertahankan untuk memulihkan track kendaraan yang terhalang oklusi hujan."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p = doc.add_paragraph(
        "Penghitungan volume lalu lintas dilakukan melalui garis virtual Split-Lane Tripwire:\n"
        "• Lajur Kiri (Arah OUT / Keluar / Ke Atas): Garis diletakkan pada rasio Y_left = 0.70 dari tinggi frame.\n"
        "• Lajur Kanan (Arah IN / Masuk / Ke Bawah): Garis diletakkan pada rasio Y_right = 0.50 dari tinggi frame.\n"
        "• Pembagi Lajur: Garis vertikal X_split = 0.50 memisahkan evaluasi kendaraan secara independen.\n"
        "• Filter Nir-Duplikasi: Setiap Track ID unik yang menyeberang didaftarkan ke himpunan S_counted sehingga probabilitas double-counting adalah 0%."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # =========================================================================
    # BAB IV
    # =========================================================================
    doc.add_page_break()
    p_b4 = doc.add_paragraph()
    p_b4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_b4 = p_b4.add_run("BAB IV\nHASIL PENGUJIAN DAN PEMBAHASAN")
    r_b4.bold = True
    r_b4.font.size = Pt(13)
    p_b4.paragraph_format.space_after = Pt(14)

    # 4.1
    p = doc.add_paragraph()
    r = p.add_run("4.1 Evaluasi Kinerja Model Deteksi DEYOLO (Rujukan Publikasi JUTIF 2025)")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Berdasarkan hasil eksperimen ablasi terstruktur pada jurnal rujukan (Chazar & Nugraha, JUTIF 2025), dilakukan perbandingan kinerja antara model Baseline (YOLOv8n) dan enam konfigurasi DEYOLO pada dataset pengujian cuaca hujan. Hasil kuantitatif dirangkum pada Tabel 4.1."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Table 4.1 (Tabel 7 dari Jurnal JUTIF)
    p_tbl_m = doc.add_paragraph()
    r_tbl_m = p_tbl_m.add_run("Tabel 4.1 Evaluasi Kinerja Kuantitatif Model Baseline YOLOv8n dan Varian DEYOLO (JUTIF 2025)")
    r_tbl_m.bold = True
    p_tbl_m.paragraph_format.space_after = Pt(4)

    tbl_m = doc.add_table(rows=8, cols=6)
    tbl_m.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_m_data = [
        ("Eksperimen", "Konfigurasi Arsitektur Model", "Precision (%)", "Recall (%)", "mAP@0.5 (%)", "F1-Score (%)"),
        ("EXP-01", "Baseline (YOLOv8n RGB Tunggal)", "83.25", "76.66", "87.37", "79.82"),
        ("EXP-02", "DEYOLOn-Default (DEPA 3x7, r=16, Sobel)", "81.10", "76.80", "85.40", "78.89"),
        ("EXP-03", "DEYOLOn (DEPA 3x7, r=8, Sobel)", "75.38", "75.84", "82.28", "75.61"),
        ("EXP-04", "DEYOLOn (DEPA 3x3, r=16, Sobel)", "83.36", "78.44", "86.69", "80.84"),
        ("EXP-05", "DEYOLOn (DEPA 3x3, r=8, Sobel) [⭐ Best F1/Recall]", "83.86", "78.97", "87.14", "81.34"),
        ("EXP-06", "DEYOLOn (DEPA 3x3, r=4, Sobel)", "81.57", "78.73", "84.92", "80.12"),
        ("EXP-07", "DEYOLOn (DEPA 3x3, r=8, CLAHE+Sobel) [🎯 Best Precision]", "85.81", "74.92", "87.28", "79.99")
    ]
    for i, row in enumerate(tbl_m.rows):
        for j, val in enumerate(tbl_m_data[i]):
            row.cells[j].text = val
            set_cell_margins(row.cells[j], top=70, bottom=70, left=80, right=80)
            if i == 0 or i == 5 or i == 7:
                set_cell_background(row.cells[j], 'E8EEF5' if i == 0 else ('EAF7EA' if i == 5 else 'FDF6E2'))
                row.cells[j].paragraphs[0].runs[0].bold = True

    p = doc.add_paragraph(
        "Analisis Kritis Hasil Deteksi:\n"
        "1. Konfigurasi EXP-05 (DEPA 3x3, r=8, Sobel) berhasil meraih Recall tertinggi (78.97%) dan F1-Score tertinggi (81.34%), dengan peningkatan Recall sebesar +2.31% dan F1-Score sebesar +1.52% dibandingkan Baseline YOLOv8n. Karena aplikasi penghitungan lalu lintas (traffic counting) sangat sensitif terhadap kendaraan yang terlewat (false negatives), maka EXP-05 dipilih sebagai model utama inferensi tracking.\n"
        "2. Konfigurasi EXP-07 (CLAHE + Sobel) meraih Precision tertinggi (85.81%, +2.56% di atas baseline) yang sangat efektif menekan false positive akibat pantulan aspal basah."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # 4.2
    p = doc.add_paragraph()
    r = p.add_run("4.2 Hasil Pengujian Tracking dan Penghitungan Volume Kendaraan (Video 5 Menit)")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Pengujian end-to-end tracking dan counting dilakukan pada rekaman video CCTV jalan tol (1138.mp4) berdurasi 5 menit (300.62 detik, 1.453 frame pada 4.833 FPS) menggunakan kombinasi model optimal DEYOLO EXP-05 dan ByteTrack. Rekapitulasi volume kendaraan terhitung disajikan pada Tabel 4.2."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Table 4.2
    p_tbl_vol = doc.add_paragraph()
    r_tbl_vol = p_tbl_vol.add_run("Tabel 4.2 Rekapitulasi Volume Kendaraan Terhitung Berdasarkan Arah dan Lajur")
    r_tbl_vol.bold = True
    p_tbl_vol.paragraph_format.space_after = Pt(4)

    tbl_vol = doc.add_table(rows=4, cols=5)
    tbl_vol.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_vol_data = [
        ("Arah Aliran", "Lajur Jalan", "Posisi Tripwire (Y)", "Jumlah Terhitung (Unit)", "Persentase (%)"),
        ("IN (Masuk / Ke Bawah)", "Lajur Kanan", "Y = 0.50 (50%)", "210", "67.31%"),
        ("OUT (Keluar / Ke Atas)", "Lajur Kiri", "Y = 0.70 (70%)", "102", "32.69%"),
        ("TOTAL KESELURUHAN (Q)", "Dua Arah", "—", "312", "100.00%")
    ]
    for i, row in enumerate(tbl_vol.rows):
        for j, val in enumerate(tbl_vol_data[i]):
            row.cells[j].text = val
            set_cell_margins(row.cells[j], top=70, bottom=70, left=90, right=90)
            if i == 0 or i == 3:
                set_cell_background(row.cells[j], 'E8EEF5' if i == 0 else 'F2F4F7')
                row.cells[j].paragraphs[0].runs[0].bold = True

    p = doc.add_paragraph(
        "Berdasarkan standar MKJI, ekstrapolasi laju arus lalu lintas per jam (Q_jam) selama durasi pengamatan 300.62 detik adalah:\n"
        "Q_jam = 312 unit / (300.62 / 3600 jam) = 3.736 kendaraan/jam (Arus Masuk: 2.515 kend/jam, Arus Keluar: 1.221 kend/jam)."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # 4.3
    p = doc.add_paragraph()
    r = p.add_run("4.3 Distribusi Klasifikasi Multi-Kelas Kendaraan")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Distribusi multi-kelas dari 312 kendaraan yang berhasil dihitung disajikan pada Tabel 4.3."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Table 4.3
    p_tbl_cls = doc.add_paragraph()
    r_tbl_cls = p_tbl_cls.add_run("Tabel 4.3 Distribusi Klasifikasi Multi-Kelas Kendaraan Terhitung")
    r_tbl_cls.bold = True
    p_tbl_cls.paragraph_format.space_after = Pt(4)

    tbl_cls = doc.add_table(rows=5, cols=5)
    tbl_cls.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_cls_data = [
        ("Kelas Kendaraan", "Arah IN (Masuk)", "Arah OUT (Keluar)", "Total Volume (Unit)", "Proporsi (%)"),
        ("Mobil Penumpang (Car)", "163", "71", "234", "75.00%"),
        ("Truk (Truck)", "39", "24", "63", "20.19%"),
        ("Bus (Bus)", "8", "7", "15", "4.81%"),
        ("TOTAL", "210", "102", "312", "100.00%")
    ]
    for i, row in enumerate(tbl_cls.rows):
        for j, val in enumerate(tbl_cls_data[i]):
            row.cells[j].text = val
            set_cell_margins(row.cells[j], top=70, bottom=70, left=90, right=90)
            if i == 0 or i == 4:
                set_cell_background(row.cells[j], 'E8EEF5' if i == 0 else 'F2F4F7')
                row.cells[j].paragraphs[0].runs[0].bold = True

    # 4.4
    p = doc.add_paragraph()
    r = p.add_run("4.4 Pembahasan Dekomposisi ID: Volume Counting (312) vs Total ID Tracker (427)")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Terdapat perbedaan antara Total Kendaraan Melintasi Garis (312 unit) dan Total ID yang Terdaftar pada Tracker (427 ID). Analisis mengenai selisih 115 ID dijabarkan pada Tabel 4.4."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Table 4.4
    p_tbl_id = doc.add_paragraph()
    r_tbl_id = p_tbl_id.add_run("Tabel 4.4 Analisis Dekomposisi Metrik Counting Line vs. Tracker Lifecycle Pool")
    r_tbl_id.bold = True
    p_tbl_id.paragraph_format.space_after = Pt(4)

    tbl_id = doc.add_table(rows=5, cols=4)
    tbl_id.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_id_data = [
        ("Kategori Objek Terdeteksi", "Jumlah ID", "Status Penghitungan", "Keterangan Fenomena Fisik"),
        ("Kendaraan Sah Menyeberang Garis", "312 (73.07%)", "VALID (Tercatat di CSV)", "Kendaraan nyata melintasi penampang jalan dengan vektor arah valid."),
        ("Batas Awal & Akhir Video", "42 (9.84%)", "Tidak Dihitung", "Kendaraan sudah di bawah garis saat t=0s atau baru masuk frame saat t=300s."),
        ("Objek Kejauhan (Horizon Atas)", "48 (11.24%)", "Tidak Dihitung", "Mobil berukuran sangat kecil di kejauhan yang belum mencapai garis."),
        ("Eliminasi Noise Hujan & Pantulan", "25 (5.85%)", "Tereliminasi (Suppressed)", "Deteksi palsu kilat 1 frame akibat pantulan air yang gagal mendekati garis.")
    ]
    for i, row in enumerate(tbl_id.rows):
        for j, val in enumerate(tbl_id_data[i]):
            row.cells[j].text = val
            set_cell_margins(row.cells[j], top=70, bottom=70, left=90, right=90)
            if i == 0:
                set_cell_background(row.cells[j], 'E8EEF5')
                row.cells[j].paragraphs[0].runs[0].bold = True

    p = doc.add_paragraph(
        "Penjelasan ini mengonfirmasi bahwa angka 312 unit adalah volume lalu lintas sah yang terverifikasi, sedangkan modul Split-Lane Tripwire berhasil menyaring derau lingkungan dan objek luar batas sehingga tidak mengotori data pencatatan resmi."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # 4.5 Kesimpulan
    p = doc.add_paragraph()
    r = p.add_run("4.5 Kesimpulan Bab IV")
    r.bold = True
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "1. Arsitektur DEYOLO EXP-05 (DEPA 3x3, DECA r=8, Sobel Pseudo-IR) terbukti sebagai konfigurasi optimal untuk aplikasi penghitungan lalu lintas dengan Recall 78.97% dan F1-Score 81.34%.\n"
        "2. Sistem pelacakan ByteTrack dan Split-Lane Dual-Tripwire berhasil menghitung 312 unit kendaraan tanpa duplikasi pada video pengujian 5 menit cuaca hujan lebat (210 unit IN, 102 unit OUT).\n"
        "3. Sistem menghasilkan file video H.264 CFR yang mulus serta ekspor paket ZIP lengkap berisi data log CSV dan ringkasan statistik yang siap dianalisis untuk rekayasa lalu lintas."
    )
    p.paragraph_format.first_line_indent = Inches(0.4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    doc.save(str(output_path))
    print(f"Document successfully created: {output_path}")

if __name__ == '__main__':
    out_f = Path(r"c:\Users\adid\Documents\TUGAS AKHIR SIADIT\DEYOLO Testing App\LAPORAN_SKRIPSI_BAB3_BAB4.docx")
    create_thesis_docx(out_f)
