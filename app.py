"""
DEYOLO Vehicle Detection & Tracking — Testing App
==================================================
Aplikasi Web interaktif (Streamlit) untuk pengujian Deteksi dan Pelacakan (Tracking)
kendaraan dalam kondisi hujan menggunakan YOLOv8 & DEYOLO (Dual-Input Pseudo-IR).

Didukung oleh algoritma Multi-Object Tracking:
1. ByteTrack (SOTA - Two-Stage IoU + Kalman Filter, Anti-ID Switching)
2. DeepSORT (Visual Re-ID Appearance + Matching Cascade)
3. Euclidean Distance Tracker (Baseline Centroid)

Dilengkapi Virtual Counting Line (Tripwire) untuk traffic counting bebas duplikat.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import shutil
import tempfile
import subprocess
import traceback
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch

from bytetrack.byte_tracker import BYTETracker, CountingLine as ByteCountingLine
from bytetrack.track_video_bytetrack import (
    get_color_for_id,
    draw_tracks as draw_byte_tracks,
    draw_hud as draw_byte_hud,
    draw_counting_line
)
from deepsort.tracker import DeepSORTTracker, CountingLine as DeepCountingLine
from tracking.euclidean_tracker import EuclideanDistTracker

# ========================= KONFIGURASI MODEL =========================
# Berdasarkan Publikasi Jurnal JUTIF (Vol. 6, No. 6, Dec 2025)
# "Dual Feature Enhancement YOLO: Spatial-Channel Attention Tuning for Vehicle Detection Under Rain Conditions"
# Penulis: Chalifa Chazar, Aditya Nugraha (Informatika ITENAS)
MODEL_REGISTRY = {
    "EXP-05 — DEYOLOn (3x3, r8, Sobel) [⭐ Best F1: 81.34% & Recall: 78.97%]": {
        "path": "weights/exp05_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "sobel",
        "desc": "Konfigurasi terbaik untuk Vehicle Counting (Recall: 78.97%, F1: 81.34%, mAP50: 87.14%). Memaksimalkan recall deteksi kendaraan saat hujan.",
    },
    "EXP-07 — DEYOLOn (3x3, r8, CLAHE+Sobel) [🎯 Best Precision: 85.81%]": {
        "path": "weights/exp07_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "clahe",
        "desc": "Konfigurasi terbaik untuk Precision (Precision: 85.81%, mAP50: 87.28%, F1: 79.99%). Mengeliminasi false positive akibat pantulan air/aspal basah.",
    },
    "EXP-02 — DEYOLOn-Default (3x7, r16, Sobel) [Original DEYOLO]": {
        "path": "weights/exp02_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "sobel",
        "desc": "Arsitektur DEYOLO asli Chen et al. (DEPA kernel 3x7, DECA r=16, Precision: 81.10%, Recall: 76.80%, mAP50: 85.40%, F1: 78.89%).",
    },
    "EXP-03 — DEYOLOn (3x7, r8, Sobel) [Stage 1 Ablation]": {
        "path": "weights/exp03_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "sobel",
        "desc": "Ablasi Tahap 1: DEPA 3x7 dengan DECA r=8 (Precision: 75.38%, Recall: 75.84%, mAP50: 82.28%, F1: 75.61%).",
    },
    "EXP-06 — DEYOLOn (3x3, r4, Sobel) [Stage 2 Ablation]": {
        "path": "weights/exp06_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "sobel",
        "desc": "Ablasi Tahap 2: DEPA 3x3 dengan DECA r=4 (Precision: 81.57%, Recall: 78.73%, mAP50: 84.92%, F1: 80.12%).",
    },
    "EXP-01 — Baseline (YOLOv8n) [Single-Branch RGB]": {
        "path": "weights/baseline_best.pt",
        "dual_input": False,
        "pseudo_ir_method": None,
        "desc": "Baseline YOLOv8n standar tanpa modul DEA dan tanpa Pseudo-IR (Precision: 83.25%, Recall: 76.66%, mAP50: 87.37%, F1: 79.82%).",
    },
}

CONF_THRESHOLD_DEFAULT = 0.25
IOU_THRESHOLD_DEFAULT = 0.70
# =====================================================================


import json

cv2.setNumThreads(os.cpu_count() or 4)


def get_exact_video_fps(video_path):
    """Mendapatkan FPS asli yang presisi dari header video menggunakan ffprobe atau cv2."""
    ffprobe_cmd = shutil.which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.EXE"
    if os.path.exists(ffprobe_cmd):
        try:
            cmd = [
                ffprobe_cmd, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames:format=duration",
                "-of", "json",
                str(video_path)
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            info = json.loads(out.decode("utf-8"))
            stream = info.get("streams", [{}])[0]
            fmt = info.get("format", {})

            for key in ["r_frame_rate", "avg_frame_rate"]:
                val = stream.get(key, "")
                if "/" in val:
                    num, den = val.split("/")
                    if float(den) > 0:
                        fps = float(num) / float(den)
                        if 0.5 <= fps <= 240.0:
                            return round(fps, 2)

            nb = stream.get("nb_frames", "")
            dur = fmt.get("duration", "")
            if nb and dur and float(dur) > 0:
                fps = float(nb) / float(dur)
                if 0.5 <= fps <= 240.0:
                    return round(fps, 2)
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and 0.5 <= fps <= 240.0:
        return round(float(fps), 2)
    return 30.0


def generate_pseudo_ir(rgb_bgr, method="sobel"):
    """Generate Pseudo-IR dari gambar RGB (format BGR/OpenCV)."""
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)

    if method == "sobel":
        blur = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=2.0, sigmaY=2.0)
    else:  # clahe
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, ksize=(5, 5), sigmaX=1.0, sigmaY=1.0)

    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    pseudo_ir = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge([pseudo_ir] * 3)


@st.cache_resource
def get_model(weights_path):
    """Load model YOLOv8 / DEYOLO dengan cache Streamlit."""
    from ultralytics import YOLO
    return YOLO(weights_path)


def run_inference(model, rgb_img, conf, iou, dual_input, pseudo_ir_img=None, imgsz=640):
    """Jalankan inferensi deteksi pada 1 frame dengan akselerasi CPU/GPU."""
    with torch.inference_mode():
        if not dual_input:
            results = model.predict(
                source=rgb_img,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                save=False,
                verbose=False
            )
        else:
            results = model.predict(
                source=[rgb_img, pseudo_ir_img],
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                save=False,
                verbose=False,
            )
    return results


def convert_to_h264(input_video_path, output_video_path, fps=30.0):
    """Konversi video ke format H.264 dengan Constant Frame Rate (CFR) agar diputar sangat mulus di browser."""
    ffmpeg_cmd = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.EXE"
    if os.path.exists(ffmpeg_cmd):
        cmd = [
            ffmpeg_cmd, "-y",
            "-i", str(input_video_path),
            "-r", f"{fps:.2f}",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "22",
            "-preset", "fast",
            "-movflags", "+faststart",
            str(output_video_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False
    return False


def create_result_zip(video_path, csv_log_path=None, csv_summary_path=None, metadata_text=""):
    """
    Membuat file ZIP in-memory (bytes) yang berisi:
    1. Video hasil tracking (.mp4)
    2. Log detail setiap kendaraan (.csv)
    3. Ringkasan statistik (.csv)
    4. Info pengujian & metadata (.txt)
    """
    import io
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if video_path and os.path.exists(str(video_path)):
            zf.write(str(video_path), arcname=f"video_{Path(video_path).name}")
        if csv_log_path and os.path.exists(str(csv_log_path)):
            zf.write(str(csv_log_path), arcname=f"data_log_kendaraan_{Path(csv_log_path).name}")
        if csv_summary_path and os.path.exists(str(csv_summary_path)):
            zf.write(str(csv_summary_path), arcname=f"ringkasan_{Path(csv_summary_path).name}")
        if metadata_text:
            zf.writestr("info_pengujian.txt", metadata_text)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ========================= STREAMLIT UI =========================
st.set_page_config(
    page_title="DEYOLO Vehicle Detection & Tracking",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 DEYOLO Vehicle Detection & Multi-Object Tracking")
st.caption("Aplikasi Pengujian Deteksi Kendaraan (Kondisi Hujan) & Pelacakan Bebas ID Switching (ByteTrack / DeepSORT)")

tab_image, tab_video = st.tabs(["📷 Deteksi Gambar Tunggal", "🎥 Pelacakan Video (Tracking & Counting)"])

# -------------------------------------------------------------
# TAB 1: DETEKSI GAMBAR TUNGGAL
# -------------------------------------------------------------
with tab_image:
    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.subheader("1. Pilih Model")
        img_model_name = st.selectbox("Model Weights:", list(MODEL_REGISTRY.keys()), key="img_model")
        img_model_cfg = MODEL_REGISTRY[img_model_name]
        st.caption(f"ℹ️ *{img_model_cfg.get('desc', '')}*")

        st.subheader("2. Upload Gambar")
        img_file = st.file_uploader("Pilih gambar input (JPG/PNG)", type=["jpg", "jpeg", "png"], key="img_upload")

        st.subheader("3. Parameter Deteksi")
        img_conf = st.slider("Confidence Threshold", 0.05, 1.0, CONF_THRESHOLD_DEFAULT, 0.05, key="img_conf")
        img_iou = st.slider("IoU Threshold (NMS)", 0.05, 1.0, IOU_THRESHOLD_DEFAULT, 0.05, key="img_iou")

        img_run_btn = st.button("🔍 Jalankan Deteksi Gambar", type="primary", use_container_width=True, key="img_run")

    with col_res:
        if img_file is not None:
            file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
            rgb_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if rgb_img is None:
                st.error("Gagal membaca gambar. Pastikan format file JPG/PNG valid.")
            else:
                if not img_run_btn:
                    st.subheader("Preview Gambar Input")
                    st.image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), caption="Gambar Input (RGB)", width="stretch")

                if img_run_btn:
                    with st.spinner("Menjalankan model..."):
                        pseudo_ir_img = None
                        pseudo_ir_display = None

                        if img_model_cfg["dual_input"]:
                            pseudo_ir_img = generate_pseudo_ir(rgb_img, method=img_model_cfg["pseudo_ir_method"])
                            pseudo_ir_display = cv2.cvtColor(pseudo_ir_img, cv2.COLOR_BGR2RGB)

                        try:
                            model = get_model(img_model_cfg["path"])
                            results = run_inference(
                                model, rgb_img, img_conf, img_iou,
                                img_model_cfg["dual_input"], pseudo_ir_img
                            )

                            result_img = results[0].plot()
                            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                            st.subheader("Hasil Deteksi")
                            if img_model_cfg["dual_input"]:
                                c1, c2, c3 = st.columns(3)
                                c1.image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), caption="RGB Original", width="stretch")
                                c2.image(pseudo_ir_display, caption=f"Pseudo-IR ({img_model_cfg['pseudo_ir_method']})", width="stretch")
                                c3.image(result_img_rgb, caption="Deteksi DEYOLO", width="stretch")
                            else:
                                st.image(result_img_rgb, caption="Hasil Deteksi Baseline YOLOv8", width="stretch")

                            st.subheader("Detail Objek Terdeteksi")
                            boxes = results[0].boxes
                            if boxes is not None and len(boxes) > 0:
                                st.success(f"Total **{len(boxes)}** objek terdeteksi.")
                                md_table = "| No | Kelas Objek | Confidence Score |\n|:---:|:---|:---:|\n"
                                for i, box in enumerate(boxes, 1):
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    cls_name = model.names[cls_id]
                                    md_table += f"| {i} | **{cls_name}** | `{conf:.3f}` |\n"
                                st.markdown(md_table)
                            else:
                                st.warning("Tidak ada objek terdeteksi pada threshold ini.")

                        except Exception as e:
                            st.error(f"Terjadi error: {e}")
                            st.code(traceback.format_exc())
        else:
            st.info("👈 Silakan upload gambar di panel kiri untuk memulai pengujian.")


# -------------------------------------------------------------
# TAB 2: PELACAKAN VIDEO (TRACKING & COUNTING)
# -------------------------------------------------------------
with tab_video:
    col_vid_ctrl, col_vid_main = st.columns([1, 2])

    with col_vid_ctrl:
        st.subheader("1. Pilih Model & Algoritma Tracking")
        vid_model_name = st.selectbox("Model Weights:", list(MODEL_REGISTRY.keys()), key="vid_model")
        vid_model_cfg = MODEL_REGISTRY[vid_model_name]
        st.caption(f"ℹ️ *{vid_model_cfg.get('desc', '')}*")

        tracker_algo = st.selectbox(
            "Algoritma Tracking:",
            [
                "ByteTrack (SOTA — Anti ID Switching) ⭐",
                "DeepSORT (Re-ID Visual Appearance)",
                "Euclidean Distance Tracker (Baseline)"
            ],
            key="vid_tracker_algo"
        )

        st.subheader("2. Input Video")
        vid_file = st.file_uploader(
            "Upload file video (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
            key="vid_upload"
        )

        st.subheader("3. Pengaturan Deteksi")
        vid_conf = st.slider("Confidence Threshold", 0.05, 1.0, CONF_THRESHOLD_DEFAULT, 0.05, key="vid_conf")
        vid_iou = st.slider("IoU Threshold (NMS)", 0.05, 1.0, IOU_THRESHOLD_DEFAULT, 0.05, key="vid_iou")

        st.subheader("4. Garis Hitung (Counting Line / Split-Lane)")
        enable_line = st.checkbox("Aktifkan Garis Hitung (0% Duplikasi)", value=True, key="vid_enable_line")
        counting_mode = st.radio(
            "Mode Garis Hitung:",
            ["Split-Lane (Lajur Kiri OUT & Kanan IN) ⭐", "Single Line (Garis Penuh)"],
            index=0,
            key="vid_counting_mode",
            help="Split-Lane memberikan posisi garis independen untuk lajur kiri (OUT) dan kanan (IN) agar mobil yang baru muncul tidak terlewat."
        )
        is_split = "Split-Lane" in counting_mode

        if is_split:
            col_la, col_lb = st.columns(2)
            with col_la:
                line_y_left = st.slider(
                    "Garis Lajur Kiri (OUT)", 0.30, 0.90, 0.70, 0.05,
                    help="Garis keluar untuk lajur kiri (arah ke atas). Diletakkan lebih ke bawah agar mobil sempat terdeteksi.",
                    key="vid_line_y_left"
                )
            with col_lb:
                line_y_right = st.slider(
                    "Garis Lajur Kanan (IN)", 0.10, 0.70, 0.50, 0.05,
                    help="Garis masuk untuk lajur kanan (arah ke bawah).",
                    key="vid_line_y_right"
                )
            split_x_ratio = st.slider("Pembagi Lajur X (Tengah Jalan)", 0.20, 0.80, 0.50, 0.05, key="vid_split_x")
            line_y_ratio = 0.55
        else:
            line_y_ratio = st.slider(
                "Posisi Garis Vertikal (Y)", 0.10, 0.90, 0.55, 0.05,
                help="Posisi garis horizontal relatif terhadap tinggi video.",
                key="vid_line_y"
            )
            line_y_left = 0.70
            line_y_right = 0.50
            split_x_ratio = 0.50

        line_direction = st.selectbox(
            "Arah Hitung:",
            ["both (Dua Arah)", "down (Masuk/Ke Bawah)", "up (Keluar/Ke Atas)"],
            key="vid_line_dir"
        )
        direction_code = "down" if "down" in line_direction else ("up" if "up" in line_direction else "both")

        st.subheader("5. Kecepatan Putar Video Hasil (Playback Speed)")
        playback_speed = st.selectbox(
            "Kecepatan Putar Video:",
            [
                "1.0x (Normal Alami Sesuai Aslinya) ⭐",
                "0.75x (Sedikit Lebih Lambat)",
                "0.5x (Slow-Motion — Rekomendasi untuk Analisis Detail)",
                "1.25x (Sedikit Lebih Cepat)",
                "1.5x (Cepat)"
            ],
            index=0,
            key="vid_speed",
            help="Pilih 1.0x untuk kecepatan normal alami, atau 0.75x / 0.5x Slow-Mo jika ingin mengamati pergerakan kendaraan dan counting dengan sangat jelas."
        )
        speed_mult = 0.5 if "0.5x" in playback_speed else (0.75 if "0.75x" in playback_speed else (1.25 if "1.25x" in playback_speed else (1.5 if "1.5x" in playback_speed else 1.0)))

        proc_mode = st.radio(
            "Mode Kelancaran Video (FPS):",
            [
                "100% Full Frame (Sangat Mulus / Smooth — Rekomendasi TA) ⭐",
                "Fast Preview (Stride 2 — Melewati 50% Frame)"
            ],
            index=0,
            key="vid_proc_mode",
            help="100% Full Frame memproses seluruh frame satu per satu tanpa ada frame yang dilewati, sehingga hasil rekaman video bergerak sangat halus/smooth sesuai FPS asli."
        )
        frame_stride = 1 if "100%" in proc_mode else 2

        vid_run_btn = st.button("🚀 Mulai Pelacakan Video", type="primary", use_container_width=True, key="vid_run")

    with col_vid_main:
        if vid_file is not None:
            temp_dir = Path(tempfile.gettempdir()) / "deyolo_tracking_all_ui"
            temp_dir.mkdir(parents=True, exist_ok=True)

            input_video_path = temp_dir / vid_file.name
            raw_output_path = temp_dir / f"tracked_raw_{vid_file.name}.mp4"
            h264_output_path = temp_dir / f"tracked_{vid_file.name}.mp4"

            # Selalu tulis buffer terbaru agar file utuh dan tidak terpotong
            with open(input_video_path, "wb") as f:
                f.write(vid_file.getbuffer())

            cap_info = cv2.VideoCapture(str(input_video_path))
            total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
            exact_fps = get_exact_video_fps(input_video_path)
            fps_input = exact_fps
            width_input = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_input = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / fps_input if fps_input > 0 else 0
            cap_info.release()

            target_playback_fps = max(1.0, min(60.0, (fps_input / frame_stride) * speed_mult))
            output_duration_sec = (total_frames / frame_stride) / target_playback_fps if target_playback_fps > 0 else 0

            dur_in_m = int(duration_sec // 60)
            dur_in_s = int(duration_sec % 60)
            dur_out_m = int(output_duration_sec // 60)
            dur_out_s = int(output_duration_sec % 60)

            st.info(
                f"📹 **Info Video**: `{vid_file.name}` | Resolusi: `{width_input}x{height_input}` | "
                f"FPS Asli: `{fps_input:.1f}` | Total: `{total_frames}` frame (Durasi Asli: `{dur_in_m:02d}:{dur_in_s:02d}`) | "
                f"Kecepatan Output: `{target_playback_fps:.1f} FPS` (Estimasi Durasi Video Hasil: `{dur_out_m:02d}:{dur_out_s:02d}`)"
            )

            if vid_run_btn:
                model = get_model(vid_model_cfg["path"])

                # Inisialisasi Tracker sesuai pilihan pengguna
                if "ByteTrack" in tracker_algo:
                    tracker = BYTETracker(track_thresh=0.40, match_thresh=0.80, track_buffer=45, frame_rate=fps_input)
                    counting_line = ByteCountingLine(
                        line_y_ratio=line_y_ratio,
                        line_y_left_ratio=line_y_left,
                        line_y_right_ratio=line_y_right,
                        split_x_ratio=split_x_ratio,
                        mode="split" if is_split else "single",
                        direction=direction_code
                    )
                elif "DeepSORT" in tracker_algo:
                    tracker = DeepSORTTracker(max_cosine_dist=0.35, max_age=40, n_init=3)
                    counting_line = DeepCountingLine(
                        line_y_ratio=line_y_ratio,
                        line_y_left_ratio=line_y_left,
                        line_y_right_ratio=line_y_right,
                        split_x_ratio=split_x_ratio,
                        mode="split" if is_split else "single",
                        direction=direction_code
                    )
                else:
                    tracker = EuclideanDistTracker(max_distance=90, max_disappeared=30)
                    counting_line = ByteCountingLine(
                        line_y_ratio=line_y_ratio,
                        line_y_left_ratio=line_y_left,
                        line_y_right_ratio=line_y_right,
                        split_x_ratio=split_x_ratio,
                        mode="split" if is_split else "single",
                        direction=direction_code
                    )

                cap = cv2.VideoCapture(str(input_video_path))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(raw_output_path),
                    fourcc,
                    target_playback_fps,
                    (width_input, height_input)
                )

                progress_bar = st.progress(0, text="Memulai tracking full video...")
                m1, m2, m3, m4 = st.columns(4)
                stat_frame = m1.empty()
                stat_fps = m2.empty()
                stat_active = m3.empty()
                stat_total = m4.empty()
                preview_caption = st.empty()
                live_preview = st.empty()

                preview_caption.caption(
                    "ℹ️ *Live preview di bawah di-refresh setiap 5 frame untuk menghemat bandwidth browser. "
                    "File video hasil akhir yang dihasilkan akan memproses 100% frame secara penuh dan sangat mulus (CFR smooth).*"
                )

                frame_idx = 0
                processed_count = 0
                consecutive_empty = 0
                max_consecutive_empty = 60  # Toleransi pemulihan jika ada frame glitch/corrupt di video
                start_time = time.time()
                prev_time = time.time()
                fps_smooth = fps_input

                is_dual = vid_model_cfg["dual_input"]
                ir_method = vid_model_cfg["pseudo_ir_method"]

                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            consecutive_empty += 1
                            if consecutive_empty > max_consecutive_empty or (total_frames > 0 and frame_idx >= total_frames):
                                # Benar-benar End of File
                                break
                            # Lewati frame glitch dan lanjutkan membaca frame berikutnya
                            frame_idx += 1
                            continue

                        consecutive_empty = 0
                        frame_idx += 1
                        if frame_stride > 1 and (frame_idx % frame_stride != 0):
                            continue

                        processed_count += 1

                        # 1. Inferensi Model dengan optimasi CPU imgsz=640
                        if is_dual:
                            pseudo_ir = generate_pseudo_ir(frame, method=ir_method)
                            results = run_inference(model, frame, vid_conf, vid_iou, True, pseudo_ir, imgsz=640)
                        else:
                            results = run_inference(model, frame, vid_conf, vid_iou, False, imgsz=640)

                        # 2. Ekstraksi Box
                        detections = []
                        if results and len(results) > 0 and results[0].boxes is not None:
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
                                cls_id = int(box.cls[0].cpu().numpy())
                                cls_name = model.names.get(cls_id, str(cls_id))
                                detections.append([x1, y1, x2, y2, conf, cls_name])

                        # 3. Update Tracker
                        if "DeepSORT" in tracker_algo:
                            tracked_objects = tracker.update(frame, detections)
                        else:
                            tracked_objects = tracker.update(detections)

                        # 4. Update Garis Hitung (Split-Lane Aware) dengan pencatatan frame & timestamp
                        count_data = counting_line.update(
                            tracked_objects,
                            height_input,
                            width_input,
                            frame_idx=frame_idx,
                            fps=fps_input
                        ) if enable_line else {
                            'total': tracker.get_total_count(), 'total_in': 0, 'total_out': 0, 'line_y': int(height_input * line_y_ratio),
                            'counted_directions': {}, 'events_count': 0
                        }

                        # 5. Hitung FPS
                        curr_time = time.time()
                        dt = curr_time - prev_time
                        prev_time = curr_time
                        if dt > 0:
                            fps_inst = 1.0 / dt
                            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst

                        # 6. Gambar visualisasi
                        display_frame = frame.copy()
                        if enable_line:
                            draw_counting_line(display_frame, count_data)
                        draw_byte_tracks(display_frame, tracked_objects, counted_info=count_data.get('counted_directions', {}))
                        draw_byte_hud(
                            display_frame,
                            fps=fps_smooth,
                            active_count=len(tracked_objects),
                            count_data=count_data,
                            model_name=Path(vid_model_cfg["path"]).stem
                        )

                        # Tulis frame penuh ke video
                        writer.write(display_frame)

                        # 7. Update UI setiap 5 frame
                        if processed_count % 5 == 0 or frame_idx >= total_frames:
                            pct = min(frame_idx / total_frames, 1.0) if total_frames > 0 else 0
                            progress_bar.progress(pct, text=f"Memproses video: {frame_idx}/{total_frames} frame ({pct*100:.1f}%)")

                            stat_frame.metric("Frame", f"{frame_idx} / {total_frames}")
                            stat_fps.metric("Processing Speed", f"{fps_smooth:.1f} FPS")
                            stat_active.metric("Active Vehicles", len(tracked_objects))
                            stat_total.metric("Garis Hitung", count_data.get('total', 0), help=f"In: {count_data.get('total_in', 0)} | Out: {count_data.get('total_out', 0)}")

                            live_preview.image(
                                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                                caption=f"Live Preview (Frame {frame_idx}/{total_frames})",
                                width="stretch"
                            )

                except Exception as e:
                    st.error(f"Error saat tracking: {e}")
                    st.code(traceback.format_exc())
                finally:
                    cap.release()
                    writer.release()

                total_elapsed = time.time() - start_time
                avg_fps = processed_count / total_elapsed if total_elapsed > 0 else 0

                progress_bar.progress(1.0, text="✅ Pemrosesan video selesai!")
                st.success(
                    f"🎉 **Pelacakan Selesai!** Total **{processed_count}** frame diproses "
                    f"dalam **{total_elapsed:.1f}s** (Kecepatan: **{avg_fps:.1f} FPS**, Output Video: **{target_playback_fps:.1f} FPS Smooth**)."
                )

                with st.spinner(f"Mengonversi video ke format H.264 CFR ({target_playback_fps:.1f} FPS)..."):
                    is_converted = convert_to_h264(raw_output_path, h264_output_path, fps=target_playback_fps)
                    final_playback_path = h264_output_path if is_converted and h264_output_path.exists() else raw_output_path

                # ==================== EKSPOR DATA CSV & ZIP ====================
                csv_log_path = temp_dir / f"vehicle_log_{Path(vid_file.name).stem}.csv"
                csv_summary_path = temp_dir / f"vehicle_summary_{Path(vid_file.name).stem}.csv"

                if enable_line:
                    df_events = counting_line.get_events_dataframe()
                    df_summary = counting_line.get_summary_dataframe()
                    counting_line.export_csv(str(csv_log_path))
                    counting_line.export_summary_csv(str(csv_summary_path))
                else:
                    import pandas as pd
                    df_events = pd.DataFrame()
                    df_summary = pd.DataFrame([
                        {'Metrik': 'Total ID Terdaftar', 'Nilai': tracker.get_total_count()}
                    ])
                    df_summary.to_csv(str(csv_summary_path), index=False)

                metadata_text = f"""DEYOLO Vehicle Tracking & Counting Report
=====================================================
Nama File Video       : {vid_file.name}
Resolusi Video        : {width_input}x{height_input}
FPS Video Asli        : {fps_input:.2f}
Target FPS Playback   : {target_playback_fps:.2f}
Total Frame Video     : {total_frames} frame
Algoritma Tracker     : {tracker_algo}
Model Deteksi         : {Path(vid_model_cfg["path"]).name}
Mode Garis Hitung     : {'Split-Lane (Dual Tripwire)' if is_split else 'Single Line'}
Total Kendaraan Masuk : {count_data.get('total_in', 0)} unit
Total Kendaraan Keluar: {count_data.get('total_out', 0)} unit
Total Keseluruhan     : {count_data.get('total', 0)} unit
Waktu Eksekusi        : {total_elapsed:.1f} detik
Tanggal Pengujian     : {time.strftime('%Y-%m-%d %H:%M:%S')}
=====================================================
"""

                zip_bytes = create_result_zip(
                    video_path=final_playback_path,
                    csv_log_path=csv_log_path if csv_log_path.exists() else None,
                    csv_summary_path=csv_summary_path if csv_summary_path.exists() else None,
                    metadata_text=metadata_text
                )

                # Player Video
                st.subheader("🎬 Hasil Pelacakan Video")
                try:
                    with open(final_playback_path, "rb") as vid_bytes:
                        st.video(vid_bytes.read())
                except Exception:
                    st.warning("Video tidak dapat diputar otomatis, namun dapat diunduh melalui tombol di bawah.")

                # Panel Download ZIP & File Terpisah
                st.subheader("📦 Download Hasil Pengujian (Video & CSV)")
                col_d1, col_d2, col_d3 = st.columns([1.6, 1.2, 1.2])

                with col_d1:
                    st.download_button(
                        label="📦 Download Paket Lengkap (.ZIP: Video + CSV Log)",
                        data=zip_bytes,
                        file_name=f"paket_hasil_tracking_{Path(vid_file.name).stem}.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True,
                        help="Berisi file video MP4, data log detail kendaraan per frame (CSV), ringkasan statistik (CSV), dan info metadata."
                    )

                with col_d2:
                    with open(final_playback_path, "rb") as vid_file_data:
                        st.download_button(
                            label="🎬 Download Video Saja (.mp4)",
                            data=vid_file_data,
                            file_name=f"tracked_{vid_file.name}",
                            mime="video/mp4",
                            use_container_width=True
                        )

                with col_d3:
                    if enable_line and csv_log_path.exists():
                        with open(csv_log_path, "rb") as f_csv:
                            st.download_button(
                                label="📊 Download Log CSV Saja (.csv)",
                                data=f_csv,
                                file_name=f"vehicle_log_{Path(vid_file.name).stem}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                # Statistik Kendaraan
                st.subheader("📊 Ringkasan Statistik Kendaraan (Bebas Duplikat)")
                s1, s2 = st.columns([1, 1])
                with s1:
                    st.metric("Total Melintasi Garis (Bebas Duplikat)", count_data.get('total', 0))
                    st.write(f"- Arah Masuk (In/Down): **{count_data.get('total_in', 0)}** unit")
                    st.write(f"- Arah Keluar (Out/Up): **{count_data.get('total_out', 0)}** unit")
                    st.write(f"- Total ID Terdaftar: **{tracker.get_total_count()}** ID")
                with s2:
                    cls_counts = count_data.get('class_counts', {}) if enable_line else tracker.get_class_counts()
                    if cls_counts:
                        md_c = "| Kelas Kendaraan | Jumlah Unit |\n|:---|:---:|\n"
                        for k, v in cls_counts.items():
                            md_c += f"| **{k}** | `{v}` |\n"
                        st.markdown(md_c)
                    else:
                        st.write("Tidak ada kendaraan yang terdeteksi.")

                # Tabel Log Detail Kendaraan
                if enable_line and not df_events.empty:
                    st.subheader(f"📋 Tabel Log Data Kendaraan Terhitung ({len(df_events)} Unit)")
                    st.caption("Data berikut tercatat otomatis saat kendaraan melintasi garis hitung virtual:")
                    st.dataframe(df_events, use_container_width=True, hide_index=True)
        else:
            st.info("👈 Upload file video di panel kiri untuk mulai pelacakan.")
