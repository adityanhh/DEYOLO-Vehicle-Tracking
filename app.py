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
MODEL_REGISTRY = {
    "EXP-05 — DEYOLOn (3x3, r8, Sobel) [Best F1]": {
        "path": "weights/exp05_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "sobel",
    },
    "EXP-07 — DEYOLOn (3x3, r8, CLAHE+Sobel) [Best Precision]": {
        "path": "weights/exp07_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "clahe",
    },
    "EXP-02 — DEYOLOn-Default (3x7, r16, Sobel)": {
        "path": "weights/exp02_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "sobel",
    },
    "EXP-03 — DEYOLOn (3x7, r16, CLAHE)": {
        "path": "weights/exp03_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "clahe",
    },
    "EXP-06 — DEYOLOn (3x3, r16, Sobel)": {
        "path": "weights/exp06_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "sobel",
    },
    "EXP-01 — Baseline (YOLOv8n)": {
        "path": "weights/baseline_best.pt",
        "dual_input": False,
        "pseudo_ir_method": None,
    },
}

CONF_THRESHOLD_DEFAULT = 0.25
IOU_THRESHOLD_DEFAULT = 0.70
# =====================================================================


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


def run_inference(model, rgb_img, conf, iou, dual_input, pseudo_ir_img=None):
    """Jalankan inferensi deteksi pada 1 frame."""
    with torch.no_grad():
        if not dual_input:
            results = model.predict(source=rgb_img, conf=conf, iou=iou, save=False, verbose=False)
        else:
            results = model.predict(
                source=[rgb_img, pseudo_ir_img],
                conf=conf,
                iou=iou,
                save=False,
                verbose=False,
            )
    return results


def convert_to_h264(input_video_path, output_video_path):
    """Konversi video ke format H.264 agar dapat diputar langsung di browser via st.video."""
    ffmpeg_cmd = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.EXE"
    if os.path.exists(ffmpeg_cmd):
        cmd = [
            ffmpeg_cmd, "-y",
            "-i", str(input_video_path),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-preset", "veryfast",
            str(output_video_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False
    return False


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

        frame_stride = st.select_slider(
            "Frame Processing Stride",
            options=[1, 2, 3],
            value=1,
            help="1 = Proses setiap frame (Sangat disarankan untuk ByteTrack agar ID stabil tanpa switch).",
            key="vid_stride"
        )

        vid_run_btn = st.button("🚀 Mulai Pelacakan Video", type="primary", use_container_width=True, key="vid_run")

    with col_vid_main:
        if vid_file is not None:
            temp_dir = Path(tempfile.gettempdir()) / "deyolo_tracking_all_ui"
            temp_dir.mkdir(parents=True, exist_ok=True)

            input_video_path = temp_dir / vid_file.name
            raw_output_path = temp_dir / f"tracked_raw_{vid_file.name}.mp4"
            h264_output_path = temp_dir / f"tracked_{vid_file.name}.mp4"

            if not input_video_path.exists() or input_video_path.stat().st_size != vid_file.size:
                with open(input_video_path, "wb") as f:
                    f.write(vid_file.getbuffer())

            cap_info = cv2.VideoCapture(str(input_video_path))
            total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_input = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
            width_input = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_input = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / fps_input if fps_input > 0 else 0
            cap_info.release()

            st.info(
                f"📹 **Info Video**: `{vid_file.name}` | Resolusi: `{width_input}x{height_input}` | "
                f"FPS: `{fps_input:.1f}` | Total: `{total_frames}` frame (~`{duration_sec/60:.1f}` menit) | "
                f"Tracker: `{tracker_algo}`"
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
                    fps_input / frame_stride,
                    (width_input, height_input)
                )

                progress_bar = st.progress(0, text="Memulai tracking...")
                m1, m2, m3, m4 = st.columns(4)
                stat_frame = m1.empty()
                stat_fps = m2.empty()
                stat_active = m3.empty()
                stat_total = m4.empty()
                live_preview = st.empty()

                frame_idx = 0
                processed_count = 0
                start_time = time.time()
                prev_time = time.time()
                fps_smooth = fps_input

                is_dual = vid_model_cfg["dual_input"]
                ir_method = vid_model_cfg["pseudo_ir_method"]

                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_idx += 1
                        if frame_stride > 1 and (frame_idx % frame_stride != 0):
                            continue

                        processed_count += 1

                        # 1. Inferensi Model
                        if is_dual:
                            pseudo_ir = generate_pseudo_ir(frame, method=ir_method)
                            results = run_inference(model, frame, vid_conf, vid_iou, True, pseudo_ir)
                        else:
                            results = run_inference(model, frame, vid_conf, vid_iou, False)

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

                        # 4. Update Garis Hitung (Split-Lane Aware)
                        count_data = counting_line.update(tracked_objects, height_input, width_input) if enable_line else {
                            'total': tracker.get_total_count(), 'total_in': 0, 'total_out': 0, 'line_y': int(height_input * line_y_ratio),
                            'counted_directions': {}
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

                        writer.write(display_frame)

                        # 7. Update UI setiap 5 frame
                        if processed_count % 5 == 0 or frame_idx >= total_frames:
                            pct = min(frame_idx / total_frames, 1.0) if total_frames > 0 else 0
                            progress_bar.progress(pct, text=f"Memproses video: {frame_idx}/{total_frames} frame ({pct*100:.1f}%)")

                            stat_frame.metric("Frame", f"{frame_idx} / {total_frames}")
                            stat_fps.metric("Processing FPS", f"{fps_smooth:.1f}")
                            stat_active.metric("Active Vehicles", len(tracked_objects))
                            stat_total.metric("Garis Hitung", count_data.get('total', 0), help=f"In: {count_data.get('total_in', 0)} | Out: {count_data.get('total_out', 0)}")

                            live_preview.image(
                                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                                caption="Live Tracking Preview (ID Tetap Konsisten Sebelum & Sesudah Garis)",
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
                    f"dalam **{total_elapsed:.1f}s** (Rata-rata: **{avg_fps:.1f} FPS**)."
                )

                with st.spinner("Mengonversi video untuk pemutar browser (H.264)..."):
                    is_converted = convert_to_h264(raw_output_path, h264_output_path)
                    final_playback_path = h264_output_path if is_converted and h264_output_path.exists() else raw_output_path

                st.subheader("🎬 Hasil Pelacakan Video")
                try:
                    with open(final_playback_path, "rb") as vid_bytes:
                        st.video(vid_bytes.read())
                except Exception:
                    st.warning("Video tidak dapat diputar otomatis, namun dapat diunduh melalui tombol di bawah.")

                with open(final_playback_path, "rb") as vid_file_data:
                    st.download_button(
                        label="📥 Download Video Hasil Tracking (.mp4)",
                        data=vid_file_data,
                        file_name=f"tracked_{vid_file.name}",
                        mime="video/mp4",
                        type="primary",
                        use_container_width=True
                    )

                st.subheader("📊 Statistik Kendaraan Unik (Bebas Duplikat)")
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
        else:
            st.info("👈 Upload file video di panel kiri untuk mulai pelacakan.")
