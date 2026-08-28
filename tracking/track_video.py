"""
Script Tracking Video Kendaraan — DEYOLO & YOLOv8
=================================================
Menjalankan deteksi + pelacakan (Euclidean Distance Tracker) pada file video
atau webcam stream, serta mengekspor video hasil pelacakan.

Contoh Penggunaan:
------------------
1. Menggunakan model DEYOLO (Dual-Input Pseudo-IR Sobel):
   python tracking/track_video.py --source video_test.mp4 --weights weights/exp05_best.pt --dual-input --pseudo-ir-method sobel --show

2. Menggunakan model Baseline (Single Input RGB):
   python tracking/track_video.py --source video_test.mp4 --weights weights/baseline_best.pt --show

3. Menyimpan hasil tracking ke file output mp4:
   python tracking/track_video.py --source video_test.mp4 --weights weights/exp05_best.pt --dual-input --output tracking_results/output.mp4
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# Pastikan root workspace terdaftar di sys.path agar modul tracking bisa di-import
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tracking.euclidean_tracker import EuclideanDistTracker


def generate_pseudo_ir(rgb_bgr, method="sobel"):
    """Generate Pseudo-IR dari gambar/frame RGB (format BGR OpenCV)."""
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


def get_color_for_id(obj_id):
    """Menghasilkan warna konsisten dan kontras untuk masing-masing ID objek."""
    palette = [
        (255, 105, 180), (0, 215, 255), (50, 205, 50), (255, 165, 0),
        (138, 43, 226), (0, 255, 255), (255, 20, 147), (30, 144, 255),
        (255, 215, 0), (0, 250, 154), (238, 130, 238), (255, 69, 0),
    ]
    return palette[obj_id % len(palette)]


def draw_hud(frame, fps, active_count, total_count, model_name):
    """Menggambar HUD (Heads-Up Display) overlay info statistik pada frame."""
    h, w = frame.shape[:2]

    # Panel background semi-transparan di pojok kiri atas
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (340, 150), (20, 20, 20), -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Garis tepi panel
    cv2.rectangle(frame, (15, 15), (340, 150), (80, 80, 80), 1)

    # Info Teks
    cv2.putText(frame, "DEYOLO Vehicle Tracker", (30, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Model: {model_name[:20]}", (30, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0) if fps >= 20 else (0, 165, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Active Vehicles: {active_count}", (30, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Total Counted: {total_count}", (30, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 200, 255), 2, cv2.LINE_AA)


def draw_tracks(frame, tracked_objects):
    """Menggambar bounding box, ID label, dan lintasan (trail) pada frame."""
    for obj in tracked_objects:
        obj_id = obj['id']
        x1, y1, x2, y2 = obj['bbox']
        cx, cy = obj['centroid']
        cls_name = obj['class_name']
        conf = obj['conf']
        trajectory = obj['trajectory']

        color = get_color_for_id(obj_id)

        # 1. Gambar jejak lintasan (trajectory line)
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                thickness = int(np.sqrt(16 * (i / len(trajectory)))) + 1
                cv2.line(frame, trajectory[i - 1], trajectory[i], color, thickness)

        # 2. Gambar Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 3. Gambar Titik Centroid
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)

        # 4. Label ID + Class + Conf
        label = f"ID:{obj_id} {cls_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_y1 = max(y1 - label_h - 8, 0)
        label_y2 = label_y1 + label_h + 8

        # Background label
        cv2.rectangle(frame, (x1, label_y1), (x1 + label_w + 10, label_y2), color, -1)
        # Teks label warna putih/hitam kontras
        cv2.putText(frame, label, (x1 + 5, label_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Tracking using DEYOLO/YOLOv8 + Euclidean Distance Tracker")
    parser.add_argument("--source", type=str, required=False, default="test_video.mp4",
                        help="Path ke file video input atau index webcam (misal '0')")
    parser.add_argument("--weights", type=str, default="weights/exp05_best.pt",
                        help="Path ke file weights model (.pt)")
    parser.add_argument("--dual-input", action="store_true", default=False,
                        help="Aktifkan mode Dual-Input DEYOLO (RGB + Pseudo-IR)")
    parser.add_argument("--pseudo-ir-method", type=str, choices=["sobel", "clahe"], default="sobel",
                        help="Metode pembuatan Pseudo-IR ('sobel' atau 'clahe')")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence score threshold deteksi (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.70,
                        help="IoU threshold NMS (default: 0.70)")
    parser.add_argument("--max-distance", type=float, default=65.0,
                        help="Jarak Euclidean maksimum antar centroid untuk matching (default: 65 px)")
    parser.add_argument("--max-disappeared", type=int, default=20,
                        help="Batas frame objek hilang sebelum ID dihapus (default: 20 frame)")
    parser.add_argument("--output", type=str, default="tracking_results/output.mp4",
                        help="Path file output video hasil tracking")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Jangan simpan file output video")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Tampilkan window video live saat proses berjalan")
    return parser.parse_args()


def run_tracking():
    args = parse_args()

    # Cek file weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] File weights tidak ditemukan: {weights_path}")
        print("Silakan periksa folder weights/ atau tentukan path yang benar via argumen --weights.")
        sys.exit(1)

    # Deteksi otomatis dual-input jika menggunakan weights exp02/exp03/exp05/exp06/exp07
    model_stem = weights_path.stem.lower()
    is_dual = args.dual_input or ("exp" in model_stem and "baseline" not in model_stem)
    if "exp07" in model_stem:
        args.pseudo_ir_method = "clahe"

    print("=" * 65)
    print("🚗 DEYOLO VEHICLE TRACKING PIPELINE (EUCLIDEAN DISTANCE)")
    print("=" * 65)
    print(f"• Input Source       : {args.source}")
    print(f"• Model Weights      : {args.weights}")
    print(f"• Mode Dual-Input    : {'Ya (RGB + Pseudo-IR)' if is_dual else 'Tidak (Single RGB)'}")
    if is_dual:
        print(f"• Pseudo-IR Method   : {args.pseudo_ir_method}")
    print(f"• Confidence Thresh  : {args.conf}")
    print(f"• Max Distance       : {args.max_distance} px")
    print(f"• Max Disappeared    : {args.max_disappeared} frames")
    print(f"• Output Video       : {args.output if not args.no_save else 'Disabled'}")
    print("=" * 65)

    # Load YOLO / DEYOLO model
    print("\n⏳ Memuat model...")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    print("✓ Model berhasil dimuat!")

    # Inisialisasi Tracker
    tracker = EuclideanDistTracker(
        max_distance=args.max_distance,
        max_disappeared=args.max_disappeared,
        trajectory_len=30
    )

    # Buka source video
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka sumber video: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"• Resolusi Video     : {width} x {height}")
    print(f"• Native FPS         : {fps_input:.1f}")
    if total_frames > 0:
        duration_sec = total_frames / fps_input
        print(f"• Total Frame        : {total_frames} frame (~{duration_sec / 60:.1f} menit)")

    # Inisialisasi Video Writer jika menyimpan video
    writer = None
    if not args.no_save:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_input, (width, height))
        print(f"• Menyiapkan output  : {output_path}")

    print("\n🚀 Memulai proses tracking... (Tekan 'q' pada window untuk berhenti, 'space' untuk jeda)")

    frame_idx = 0
    start_time_all = time.time()
    prev_time = time.time()
    fps_smooth = fps_input
    paused = False

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # 1. Preprocessing input sesuai tipe model
                if is_dual:
                    pseudo_ir = generate_pseudo_ir(frame, method=args.pseudo_ir_method)
                    with torch.no_grad():
                        results = model.predict(
                            source=[frame, pseudo_ir],
                            conf=args.conf,
                            iou=args.iou,
                            save=False,
                            verbose=False
                        )
                else:
                    with torch.no_grad():
                        results = model.predict(
                            source=frame,
                            conf=args.conf,
                            iou=args.iou,
                            save=False,
                            verbose=False
                        )

                # 2. Ekstraksi bounding boxes hasil deteksi
                detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = model.names.get(cls_id, str(cls_id))
                        detections.append([x1, y1, x2, y2, conf, cls_name])

                # 3. Update Euclidean Distance Tracker
                tracked_objects = tracker.update(detections)

                # 4. Hitung FPS aktual
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps_inst = 1.0 / dt
                    fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst

                # 5. Gambar hasil visualisasi
                display_frame = frame.copy()
                draw_tracks(display_frame, tracked_objects)
                draw_hud(
                    display_frame,
                    fps=fps_smooth,
                    active_count=len(tracked_objects),
                    total_count=tracker.get_total_count(),
                    model_name=weights_path.stem
                )

                # 6. Tulis ke file output video
                if writer is not None:
                    writer.write(display_frame)

                # 7. Print progress ke console secara periodik
                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    print(
                        f"\r[Frame {frame_idx:5d}/{total_frames if total_frames > 0 else '?'}] "
                        f"({pct:5.1f}%) | "
                        f"FPS: {fps_smooth:4.1f} | "
                        f"Active: {len(tracked_objects):2d} | "
                        f"Total Unique: {tracker.get_total_count():3d}",
                        end="", flush=True
                    )

            # 8. Tampilkan window jika flag --show aktif
            if args.show:
                cv2.imshow("DEYOLO Vehicle Tracking (Euclidean Distance)", display_frame)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key == ord('q') or key == 27:  # q atau ESC
                    print("\n[INFO] Dihentikan oleh pengguna.")
                    break
                elif key == ord(' '):  # Spasi untuk Pause/Resume
                    paused = not paused
                    state = "PAUSED" if paused else "RESUMED"
                    print(f"\n[INFO] {state}")

    except KeyboardInterrupt:
        print("\n[INFO] Proses dihentikan dengan KeyboardInterrupt.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    total_elapsed = time.time() - start_time_all
    avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0

    print("\n\n" + "=" * 65)
    print("🎉 PROSES TRACKING SELESAI")
    print("=" * 65)
    print(f"• Total Frame Diproses : {frame_idx} frame")
    print(f"• Total Waktu Eksekusi : {total_elapsed:.2f} detik")
    print(f"• Rata-rata FPS        : {avg_fps:.2f} FPS")
    print(f"• Total Kendaraan Unik : {tracker.get_total_count()} kendaraan")
    print("• Rincian per Kelas    :")
    for cls_name, count in tracker.get_class_counts().items():
        print(f"    - {cls_name:15s}: {count} unit")

    if not args.no_save:
        print(f"• Video Tersimpan Di   : {Path(args.output).resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    run_tracking()
