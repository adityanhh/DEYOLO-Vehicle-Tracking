"""
ByteTrack Vehicle Tracking & Counting — Dedicated Streamlit App
================================================================
Aplikasi Web UI khusus pelacakan kendaraan SOTA menggunakan ByteTrack
dan Virtual Counting Line (Tripwire) untuk traffic monitoring bebas duplikasi.

Cara jalankan:
    streamlit run bytetrack/bytetrack_app.py
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
import torch

# Pastikan root workspace terdaftar di sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bytetrack.byte_tracker import BYTETracker, CountingLine
from bytetrack.track_video_bytetrack import get_color_for_id, draw_tracks, draw_hud, draw_counting_line

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
# =====================================================================


def generate_pseudo_ir(rgb_bgr, method="sobel"):
    """Generate Pseudo-IR dari frame RGB."""
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
    """Load model dengan cache Streamlit."""
    from ultralytics import YOLO
    return YOLO(weights_path)


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
                "-show_entries", "stream=r_frame_rate,avg_frame_rate",
                "-of", "json",
                str(video_path)
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            info = json.loads(out.decode("utf-8"))
            stream = info.get("streams", [{}])[0]
            for key in ["avg_frame_rate", "r_frame_rate"]:
                val = stream.get(key, "")
                if "/" in val:
                    num, den = val.split("/")
                    if float(den) > 0:
                        fps = float(num) / float(den)
                        if 5.0 <= fps <= 120.0:
                            return round(fps, 2)
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and 5.0 <= fps <= 120.0:
        return round(float(fps), 2)
    return 30.0


def run_inference(model, rgb_img, conf, iou, dual_input, pseudo_ir_img=None, imgsz=640):
    """Jalankan inferensi deteksi pada 1 frame dengan akselerasi tensor imgsz=640."""
    with torch.inference_mode():
        if not dual_input:
            results = model.predict(source=rgb_img, conf=conf, iou=iou, imgsz=imgsz, save=False, verbose=False)
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


# ========================= STREAMLIT UI =========================
st.set_page_config(
    page_title="ByteTrack Vehicle Tracking & Counting",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 ByteTrack Vehicle Tracking & Counting (DEYOLO)")
st.caption("SOTA Multi-Object Tracking (Kalman Filter + Two-Stage IoU Matching) & Virtual Counting Line")

col_ctrl, col_main = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Pilih Model Weights")
    selected_model_name = st.selectbox("Model:", list(MODEL_REGISTRY.keys()))
    model_cfg = MODEL_REGISTRY[selected_model_name]

    st.subheader("2. Upload Video")
    uploaded_video = st.file_uploader(
        "Upload Video (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="byte_video_upload"
    )

    st.subheader("3. Pengaturan Deteksi")
    conf_thresh = st.slider("Min Confidence Detection", 0.05, 1.0, 0.20, 0.05,
                            help="Threshold confidence awal. Deteksi low-score akan dipulihkan oleh ByteTrack.")
    iou_thresh = st.slider("NMS IoU Threshold", 0.05, 1.0, 0.70, 0.05)

    st.subheader("4. Pengaturan ByteTrack")
    track_thresh = st.slider(
        "Track High-Score Threshold", 0.20, 0.90, 0.45, 0.05,
        help="Batas pemisah antara High-Score Detection (Tahap 1) dan Low-Score Detection (Tahap 2)."
    )
    match_thresh = st.slider(
        "IoU Matching Threshold", 0.40, 0.95, 0.80, 0.05,
        help="Batas toleransi IoU distance antara prediksi Kalman Filter dan deteksi."
    )
    track_buffer = st.slider(
        "Track Buffer (Max Lost Frames)", 10, 100, 30, 5,
        help="Jumlah frame track disimpan saat kendaraan hilang (misal tertutup hujan/wiper) sebelum dihapus."
    )

    st.subheader("5. Garis Hitung (Counting Line / Split-Lane)")
    enable_line = st.checkbox("Aktifkan Garis Hitung (0% Duplikasi)", value=True)
    counting_mode = st.radio(
        "Mode Garis Hitung:",
        ["Split-Lane (Lajur Kiri OUT & Kanan IN) ⭐", "Single Line (Garis Penuh)"],
        index=0,
        help="Split-Lane memberikan posisi garis independen untuk lajur kiri (OUT) dan kanan (IN) agar mobil yang baru muncul tidak terlewat."
    )
    is_split = "Split-Lane" in counting_mode

    if is_split:
        col_la, col_lb = st.columns(2)
        with col_la:
            line_y_left = st.slider(
                "Garis Lajur Kiri (OUT)", 0.30, 0.90, 0.70, 0.05,
                help="Garis keluar untuk lajur kiri (arah ke atas). Diletakkan lebih ke bawah agar mobil sempat terdeteksi."
            )
        with col_lb:
            line_y_right = st.slider(
                "Garis Lajur Kanan (IN)", 0.10, 0.70, 0.50, 0.05,
                help="Garis masuk untuk lajur kanan (arah ke bawah)."
            )
        split_x_ratio = st.slider("Pembagi Lajur X (Tengah Jalan)", 0.20, 0.80, 0.50, 0.05)
        line_y_ratio = 0.55
    else:
        line_y_ratio = st.slider(
            "Posisi Garis Vertikal (Y)", 0.10, 0.90, 0.55, 0.05,
            help="Posisi garis horizontal relatif terhadap tinggi video."
        )
        line_y_left = 0.70
        line_y_right = 0.50
        split_x_ratio = 0.50

    line_direction = st.selectbox("Arah yang Dihitung:", ["both (Dua Arah)", "down (Masuk/Ke Bawah)", "up (Keluar/Ke Atas)"])
    direction_code = "down" if "down" in line_direction else ("up" if "up" in line_direction else "both")

    st.subheader("6. Kecepatan Putar Video Hasil (Playback Speed)")
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
        help="100% Full Frame memproses seluruh frame tanpa ada yang dilewati untuk hasil video yang halus dan tracking konsisten."
    )
    frame_stride = 1 if "100%" in proc_mode else 2

    run_button = st.button("🚀 Mulai ByteTrack Video", type="primary", use_container_width=True)

with col_main:
    if uploaded_video is not None:
        temp_dir = Path(tempfile.gettempdir()) / "deyolo_bytetrack_ui"
        temp_dir.mkdir(parents=True, exist_ok=True)

        input_path = temp_dir / uploaded_video.name
        raw_output_path = temp_dir / f"bytetrack_raw_{uploaded_video.name}.mp4"
        h264_output_path = temp_dir / f"bytetrack_{uploaded_video.name}.mp4"

        if not input_path.exists() or input_path.stat().st_size != uploaded_video.size:
            with open(input_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

        cap_info = cv2.VideoCapture(str(input_path))
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        exact_fps = get_exact_video_fps(input_path)
        fps_input = exact_fps
        width_input = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_input = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps_input if fps_input > 0 else 0
        cap_info.release()

        target_playback_fps = max(5.0, min(60.0, (fps_input / frame_stride) * speed_mult))

        st.info(
            f"📹 **Video Input**: `{uploaded_video.name}` | Resolusi: `{width_input}x{height_input}` | "
            f"FPS Asli: `{fps_input:.1f}` | Total: `{total_frames}` frame (~`{duration_sec/60:.1f}` menit) | "
            f"Kecepatan Output: `{target_playback_fps:.1f} FPS ({playback_speed.split(' ')[0]})`"
        )

        if run_button:
            model = get_model(model_cfg["path"])
            tracker = BYTETracker(
                track_thresh=track_thresh,
                match_thresh=match_thresh,
                track_buffer=track_buffer,
                frame_rate=fps_input,
                trajectory_len=30
            )
            counting_line = CountingLine(
                line_y_ratio=line_y_ratio,
                line_y_left_ratio=line_y_left,
                line_y_right_ratio=line_y_right,
                split_x_ratio=split_x_ratio,
                mode="split" if is_split else "single",
                direction=direction_code
            )

            cap = cv2.VideoCapture(str(input_path))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(raw_output_path),
                fourcc,
                target_playback_fps,
                (width_input, height_input)
            )

            progress_bar = st.progress(0, text="Menginisialisasi ByteTrack...")
            m1, m2, m3, m4 = st.columns(4)
            stat_frame = m1.empty()
            stat_fps = m2.empty()
            stat_active = m3.empty()
            stat_line = m4.empty()
            preview_caption = st.empty()
            live_preview = st.empty()

            preview_caption.caption(
                "ℹ️ *Live preview di-refresh berkala untuk performa web browser. "
                "File video hasil akhir akan memproses seluruh frame secara penuh dan sangat mulus (CFR 30/60 FPS asli).*"
            )

            frame_idx = 0
            processed_count = 0
            start_time = time.time()
            prev_time = time.time()
            fps_smooth = fps_input

            is_dual = model_cfg["dual_input"]
            ir_method = model_cfg["pseudo_ir_method"]

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_stride > 1 and (frame_idx % frame_stride != 0):
                        continue

                    processed_count += 1

                    # 1. Inferensi Model dengan imgsz=640
                    if is_dual:
                        pseudo_ir = generate_pseudo_ir(frame, method=ir_method)
                        results = run_inference(model, frame, conf_thresh, iou_thresh, True, pseudo_ir, imgsz=640)
                    else:
                        results = run_inference(model, frame, conf_thresh, iou_thresh, False, imgsz=640)

                    # 2. Ekstraksi Box
                    detections = []
                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = model.names.get(cls_id, str(cls_id))
                            detections.append([x1, y1, x2, y2, conf, cls_name])

                    # 3. Update ByteTrack
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

                    # 6. Render Visual
                    display_frame = frame.copy()
                    if enable_line:
                        draw_counting_line(display_frame, count_data)
                    draw_tracks(display_frame, tracked_objects, counted_info=count_data.get('counted_directions', {}))
                    draw_hud(
                        display_frame,
                        fps=fps_smooth,
                        active_count=len(tracked_objects),
                        count_data=count_data,
                        model_name=Path(model_cfg["path"]).stem
                    )

                    writer.write(display_frame)

                    # 7. Update UI setiap 5 frame
                    if processed_count % 5 == 0 or frame_idx >= total_frames:
                        pct = min(frame_idx / total_frames, 1.0) if total_frames > 0 else 0
                        progress_bar.progress(pct, text=f"ByteTrack Memproses: {frame_idx}/{total_frames} frame ({pct*100:.1f}%)")

                        stat_frame.metric("Frame", f"{frame_idx} / {total_frames}")
                        stat_fps.metric("Processing Speed", f"{fps_smooth:.1f} FPS")
                        stat_active.metric("Active Vehicles", len(tracked_objects))
                        stat_line.metric("Garis Hitung", count_data.get('total', 0), help=f"In: {count_data.get('total_in', 0)} | Out: {count_data.get('total_out', 0)}")

                        live_preview.image(
                            cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                            caption=f"Live ByteTrack Preview (Frame {frame_idx}/{total_frames})",
                            width="stretch"
                        )

            except Exception as e:
                st.error(f"Error saat menjalankan ByteTrack: {e}")
                st.code(traceback.format_exc())
            finally:
                cap.release()
                writer.release()

            total_elapsed = time.time() - start_time
            avg_fps = processed_count / total_elapsed if total_elapsed > 0 else 0

            progress_bar.progress(1.0, text="✅ Pemrosesan ByteTrack Selesai!")
            st.success(
                f"🎉 **ByteTrack Selesai!** Total **{processed_count}** frame diproses "
                f"dalam **{total_elapsed:.1f}s** (Kecepatan: **{avg_fps:.1f} FPS**, Output Video: **{target_playback_fps:.1f} FPS Smooth**)."
            )

            with st.spinner(f"Mengonversi video ke format H.264 CFR ({target_playback_fps:.1f} FPS)..."):
                is_converted = convert_to_h264(raw_output_path, h264_output_path, fps=target_playback_fps)
                final_playback_path = h264_output_path if is_converted and h264_output_path.exists() else raw_output_path

            # Player Video
            st.subheader("🎬 Hasil Pelacakan Video (ByteTrack)")
            try:
                with open(final_playback_path, "rb") as vid_bytes:
                    st.video(vid_bytes.read())
            except Exception:
                st.warning("Video tidak dapat diputar otomatis, silakan gunakan tombol download.")

            # Tombol Download
            with open(final_playback_path, "rb") as vid_file_data:
                st.download_button(
                    label="📥 Download Video ByteTrack (.mp4)",
                    data=vid_file_data,
                    file_name=f"bytetrack_{uploaded_video.name}",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True
                )

            # Statistik & Evaluasi
            st.subheader("📊 Ringkasan Statistik Kendaraan")
            s1, s2 = st.columns(2)
            with s1:
                st.metric("Total Melintasi Garis (Bebas Duplikat)", count_data.get('total', 0))
                st.write(f"- Arah Masuk (In/Down): **{count_data.get('total_in', 0)}** unit")
                st.write(f"- Arah Keluar (Out/Up): **{count_data.get('total_out', 0)}** unit")
                st.write(f"- Total ID ByteTrack Terdaftar: **{tracker.get_total_count()}** ID")
            with s2:
                st.write("**Distribusi per Kelas Kendaraan:**")
                cls_counts = count_data.get('class_counts', {}) if enable_line else tracker.get_class_counts()
                if cls_counts:
                    md_c = "| Kelas Kendaraan | Jumlah Unit |\n|:---|:---:|\n"
                    for k, v in cls_counts.items():
                        md_c += f"| **{k}** | `{v}` |\n"
                    st.markdown(md_c)
                else:
                    st.write("Tidak ada kendaraan yang terhitung.")
    else:
        st.info("👈 Upload file video di panel kiri untuk mulai pelacakan dengan ByteTrack.")
