"""
DeepSORT Vehicle Tracking & Counting — Dedicated Streamlit App
===============================================================
Aplikasi Web UI khusus pelacakan kendaraan SOTA menggunakan DeepSORT
(Deep Appearance Re-ID + Matching Cascade) dan Virtual Counting Line (Tripwire).

Cara jalankan:
    streamlit run deepsort/deepsort_app.py
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

from deepsort.tracker import DeepSORTTracker, CountingLine
from deepsort.track_video_deepsort import get_color_for_id, draw_tracks, draw_hud, draw_counting_line

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
    """Konversi video ke format H.264 agar dapat diputar langsung di browser."""
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
    page_title="DeepSORT Vehicle Tracking & Counting",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 DeepSORT Vehicle Tracking & Counting (DEYOLO)")
st.caption("SOTA Multi-Object Tracking (Deep Appearance Re-ID + Matching Cascade + Kalman Filter)")

col_ctrl, col_main = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Pilih Model Weights")
    selected_model_name = st.selectbox("Model:", list(MODEL_REGISTRY.keys()))
    model_cfg = MODEL_REGISTRY[selected_model_name]

    st.subheader("2. Upload Video")
    uploaded_video = st.file_uploader(
        "Upload Video (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="deepsort_video_upload"
    )

    st.subheader("3. Pengaturan Deteksi")
    conf_thresh = st.slider("Confidence Detection", 0.05, 1.0, 0.25, 0.05)
    iou_thresh = st.slider("NMS IoU Threshold", 0.05, 1.0, 0.70, 0.05)

    st.subheader("4. Pengaturan DeepSORT")
    max_cosine_dist = st.slider(
        "Max Cosine Distance (Re-ID Appearance)", 0.10, 0.80, 0.35, 0.05,
        help="Batas jarak kosinus kemiripan visual. Nilai lebih kecil = lebih ketat/selektif."
    )
    max_age = st.slider(
        "Max Age (Frame Buffer)", 10, 100, 30, 5,
        help="Jumlah frame track disimpan saat kendaraan hilang sebelum dihapus."
    )
    n_init = st.slider(
        "N-Init Confirmation", 1, 10, 3, 1,
        help="Jumlah frame konfirmasi sebelum track berstatus Confirmed."
    )

    st.subheader("5. Pengaturan Garis Hitung (Counting Line)")
    enable_line = st.checkbox("Aktifkan Garis Hitung (Tripwire)", value=True)
    line_y_ratio = st.slider(
        "Posisi Garis Vertikal (Y)", 0.10, 0.90, 0.55, 0.05,
        help="Posisi garis horizontal relatif terhadap tinggi video."
    )
    line_direction = st.selectbox("Arah yang Dihitung:", ["both (Dua Arah)", "down (Masuk/Ke Bawah)", "up (Keluar/Ke Atas)"])
    direction_code = "down" if "down" in line_direction else ("up" if "up" in line_direction else "both")

    frame_stride = st.select_slider(
        "Frame Stride",
        options=[1, 2, 3],
        value=1,
        help="1 = Proses setiap frame (Direkomendasikan agar Kalman Filter dan Re-ID presisi)."
    )

    run_button = st.button("🚀 Mulai DeepSORT Video", type="primary", use_container_width=True)

with col_main:
    if uploaded_video is not None:
        temp_dir = Path(tempfile.gettempdir()) / "deyolo_deepsort_ui"
        temp_dir.mkdir(parents=True, exist_ok=True)

        input_path = temp_dir / uploaded_video.name
        raw_output_path = temp_dir / f"deepsort_raw_{uploaded_video.name}.mp4"
        h264_output_path = temp_dir / f"deepsort_{uploaded_video.name}.mp4"

        if not input_path.exists() or input_path.stat().st_size != uploaded_video.size:
            with open(input_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

        cap_info = cv2.VideoCapture(str(input_path))
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_input = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
        width_input = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_input = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps_input if fps_input > 0 else 0
        cap_info.release()

        st.info(
            f"📹 **Video Input**: `{uploaded_video.name}` | Resolusi: `{width_input}x{height_input}` | "
            f"FPS: `{fps_input:.1f}` | Total: `{total_frames}` frame (~`{duration_sec/60:.1f}` menit)"
        )

        if run_button:
            model = get_model(model_cfg["path"])
            tracker = DeepSORTTracker(
                max_cosine_dist=max_cosine_dist,
                max_age=max_age,
                n_init=n_init,
                trajectory_len=30
            )
            counting_line = CountingLine(line_y_ratio=line_y_ratio, direction=direction_code)

            cap = cv2.VideoCapture(str(input_path))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(raw_output_path),
                fourcc,
                fps_input / frame_stride,
                (width_input, height_input)
            )

            progress_bar = st.progress(0, text="Menginisialisasi DeepSORT & Re-ID...")
            m1, m2, m3, m4 = st.columns(4)
            stat_frame = m1.empty()
            stat_fps = m2.empty()
            stat_active = m3.empty()
            stat_line = m4.empty()
            live_preview = st.empty()

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

                    # 1. Inferensi Model
                    if is_dual:
                        pseudo_ir = generate_pseudo_ir(frame, method=ir_method)
                        results = run_inference(model, frame, conf_thresh, iou_thresh, True, pseudo_ir)
                    else:
                        results = run_inference(model, frame, conf_thresh, iou_thresh, False)

                    # 2. Ekstraksi Box
                    detections = []
                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = model.names.get(cls_id, str(cls_id))
                            detections.append([x1, y1, x2, y2, conf, cls_name])

                    # 3. Update DeepSORT (Feature Extraction + Cascade Matching + Kalman)
                    tracked_objects = tracker.update(frame, detections)

                    # 4. Update Garis Hitung
                    count_data = counting_line.update(tracked_objects, height_input) if enable_line else {
                        'total': tracker.get_total_count(), 'total_in': 0, 'total_out': 0, 'line_y': int(height_input * line_y_ratio)
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
                        draw_counting_line(display_frame, count_data['line_y'])
                    draw_tracks(display_frame, tracked_objects)
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
                        progress_bar.progress(pct, text=f"DeepSORT Memproses: {frame_idx}/{total_frames} frame ({pct*100:.1f}%)")

                        stat_frame.metric("Frame", f"{frame_idx} / {total_frames}")
                        stat_fps.metric("Processing FPS", f"{fps_smooth:.1f}")
                        stat_active.metric("Active Vehicles", len(tracked_objects))
                        stat_line.metric("Garis Hitung", count_data.get('total', 0), help=f"In: {count_data.get('total_in', 0)} | Out: {count_data.get('total_out', 0)}")

                        live_preview.image(
                            cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                            caption="Live DeepSORT Preview",
                            width="stretch"
                        )

            except Exception as e:
                st.error(f"Error saat menjalankan DeepSORT: {e}")
                st.code(traceback.format_exc())
            finally:
                cap.release()
                writer.release()

            total_elapsed = time.time() - start_time
            avg_fps = processed_count / total_elapsed if total_elapsed > 0 else 0

            progress_bar.progress(1.0, text="✅ Pemrosesan DeepSORT Selesai!")
            st.success(
                f"🎉 **DeepSORT Selesai!** Total **{processed_count}** frame diproses "
                f"dalam **{total_elapsed:.1f}s** (Rata-rata: **{avg_fps:.1f} FPS**)."
            )

            with st.spinner("Mengonversi video ke H.264 untuk web browser..."):
                is_converted = convert_to_h264(raw_output_path, h264_output_path)
                final_playback_path = h264_output_path if is_converted and h264_output_path.exists() else raw_output_path

            # Player Video
            st.subheader("🎬 Hasil Pelacakan Video (DeepSORT)")
            try:
                with open(final_playback_path, "rb") as vid_bytes:
                    st.video(vid_bytes.read())
            except Exception:
                st.warning("Video tidak dapat diputar otomatis, silakan gunakan tombol download.")

            # Tombol Download
            with open(final_playback_path, "rb") as vid_file_data:
                st.download_button(
                    label="📥 Download Video DeepSORT (.mp4)",
                    data=vid_file_data,
                    file_name=f"deepsort_{uploaded_video.name}",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True
                )

            # Statistik & Evaluasi
            st.subheader("📊 Ringkasan Statistik Kendaraan (DeepSORT)")
            s1, s2 = st.columns(2)
            with s1:
                st.metric("Total Melintasi Garis (Bebas Duplikat)", count_data.get('total', 0))
                st.write(f"- Arah Masuk (In/Down): **{count_data.get('total_in', 0)}** unit")
                st.write(f"- Arah Keluar (Out/Up): **{count_data.get('total_out', 0)}** unit")
                st.write(f"- Total ID DeepSORT Terkonfirmasi: **{tracker.get_total_count()}** ID")
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
        st.info("👈 Upload file video di panel kiri untuk mulai pelacakan dengan DeepSORT.")
