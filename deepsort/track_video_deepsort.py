"""
Script Tracking Video — DeepSORT SOTA + DEYOLO
===============================================
Menjalankan pelacakan kendaraan dengan algoritma DeepSORT (Re-ID Appearance Descriptor +
Matching Cascade + Kalman Filter) dan Virtual Counting Line untuk traffic counting bebas duplikat.

Contoh Penggunaan:
------------------
1. Menggunakan model DEYOLO (Dual-Input Pseudo-IR Sobel):
   python deepsort/track_video_deepsort.py --source video_test.mp4 --weights weights/exp05_best.pt --show

2. Menggunakan Counting Line (Garis Hitung pada 55% tinggi layar):
   python deepsort/track_video_deepsort.py --source video_test.mp4 --weights weights/exp05_best.pt --line-y 0.55 --show

3. Menyimpan hasil tracking ke file output mp4:
   python deepsort/track_video_deepsort.py --source video_test.mp4 --weights weights/exp05_best.pt --output deepsort_results/hasil_deepsort.mp4
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# Daftarkan root workspace ke sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepsort.tracker import DeepSORTTracker, CountingLine


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


def get_color_for_id(obj_id):
    """Menghasilkan warna konsisten per ID objek."""
    palette = [
        (255, 105, 180), (0, 215, 255), (50, 205, 50), (255, 165, 0),
        (138, 43, 226), (0, 255, 255), (255, 20, 147), (30, 144, 255),
        (255, 215, 0), (0, 250, 154), (238, 130, 238), (255, 69, 0),
    ]
    return palette[obj_id % len(palette)]


def draw_hud(frame, fps, active_count, count_data, model_name):
    """Menggambar HUD overlay DeepSORT pada frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (370, 165), (20, 20, 20), -1)
    alpha = 0.70
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.rectangle(frame, (15, 15), (370, 165), (100, 100, 100), 1)

    cv2.putText(frame, "DeepSORT Vehicle Tracker", (25, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Model: {model_name[:22]}", (25, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (25, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0) if fps >= 20 else (0, 165, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Active Vehicles: {active_count}", (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    line_total = count_data.get('total', 0)
    line_in = count_data.get('total_in', 0)
    line_out = count_data.get('total_out', 0)
    cv2.putText(frame, f"Line Counted: {line_total} (In:{line_in} | Out:{line_out})", (25, 142),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 220, 255), 2, cv2.LINE_AA)


def draw_counting_line(frame, count_data):
    """
    Menggambar garis virtual tripwire pada frame (Mendukung Split-Lane dan Single Line).
    """
    h, w = frame.shape[:2]

    if isinstance(count_data, int):
        line_y = count_data
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(frame, (0, line_y), (w, line_y), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"COUNTING LINE (Y={line_y})", (w - 240, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        return

    mode = count_data.get('mode', 'split')
    if mode == "split":
        split_x = count_data.get('split_x', w // 2)
        line_y_left = count_data.get('line_y_left', int(h * 0.70))
        line_y_right = count_data.get('line_y_right', int(h * 0.50))

        # 1. Garis Pembagi Lajur (Dashed Vertical Divider)
        y_div_start = max(0, min(line_y_left, line_y_right) - 60)
        y_div_end = min(h, max(line_y_left, line_y_right) + 60)
        for y_dash in range(y_div_start, y_div_end, 16):
            cv2.line(frame, (split_x, y_dash), (split_x, min(y_dash + 8, y_div_end)), (200, 200, 200), 1, cv2.LINE_AA)

        # 2. Garis Lajur Kiri (Arah OUT / Ke Atas) - Warna Neon Orange/Coral
        cv2.line(frame, (0, line_y_left), (split_x - 5, line_y_left), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(frame, (0, line_y_left), (split_x - 5, line_y_left), (50, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"LANE OUT (Y={line_y_left})", (20, line_y_left - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 140, 255), 1, cv2.LINE_AA)

        # 3. Garis Lajur Kanan (Arah IN / Ke Bawah) - Warna Neon Cyan
        cv2.line(frame, (split_x + 5, line_y_right), (w, line_y_right), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(frame, (split_x + 5, line_y_right), (w, line_y_right), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"LANE IN (Y={line_y_right})", (w - 200, line_y_right - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    else:
        line_y = count_data.get('line_y', int(h * 0.55))
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(frame, (0, line_y), (w, line_y), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"COUNTING LINE (Y={line_y})", (w - 240, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)


def draw_tracks(frame, tracked_objects, counted_info=None):
    """
    Menggambar bounding box, label ID, centroid, dan lintasan trajectory.
    ID pelacakan tetap konstan dan konsisten sebelum dan sesudah melewati garis counting.
    """
    if counted_info is None:
        counted_info = {}

    for obj in tracked_objects:
        obj_id = obj['id']
        x1, y1, x2, y2 = obj['bbox']
        cx, cy = obj['centroid']
        cls_name = obj['class_name']
        conf = obj['conf']
        trajectory = obj['trajectory']

        color = get_color_for_id(obj_id)
        is_counted = obj_id in counted_info
        direction_tag = f" [{counted_info[obj_id]}]" if is_counted else ""

        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                thickness = int(np.sqrt(16 * (i / len(trajectory)))) + 1
                cv2.line(frame, trajectory[i - 1], trajectory[i], color, thickness)

        box_thickness = 3 if is_counted else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)

        label = f"ID:{obj_id} {cls_name}{direction_tag} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

        label_y1 = max(y1 - label_h - 8, 0)
        label_y2 = label_y1 + label_h + 8

        cv2.rectangle(frame, (x1, label_y1), (x1 + label_w + 10, label_y2), color, -1)
        cv2.putText(frame, label, (x1 + 5, label_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSORT Vehicle Tracking with DEYOLO / YOLOv8")
    parser.add_argument("--source", type=str, default="test_video.mp4",
                        help="Path ke file video input atau index webcam ('0')")
    parser.add_argument("--weights", type=str, default="weights/exp05_best.pt",
                        help="Path ke weights model YOLO (.pt)")
    parser.add_argument("--dual-input", action="store_true", default=False,
                        help="Aktifkan Dual-Input DEYOLO (RGB + Pseudo-IR)")
    parser.add_argument("--pseudo-ir-method", type=str, choices=["sobel", "clahe"], default="sobel",
                        help="Metode pembuatan Pseudo-IR ('sobel' atau 'clahe')")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Minimum detection confidence score (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.70,
                        help="NMS IoU threshold (default: 0.70)")
    parser.add_argument("--max-cosine-dist", type=float, default=0.35,
                        help="Threshold jarak kosinus penampilan visual Re-ID (default: 0.35)")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Jumlah frame simpan track yang hilang sebelum dihapus (default: 30)")
    parser.add_argument("--n-init", type=int, default=3,
                        help="Jumlah frame konfirmasi sebelum track aktif (default: 3)")
    parser.add_argument("--line-y", type=float, default=0.55,
                        help="Posisi vertikal Counting Line (rasio 0.0 - 1.0, default: 0.55)")
    parser.add_argument("--output", type=str, default="deepsort_results/output.mp4",
                        help="Path file output video hasil tracking")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Jangan simpan file output video")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Tampilkan window video live saat proses berjalan")
    return parser.parse_args()


def run_deepsort():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] File weights tidak ditemukan: {weights_path}")
        sys.exit(1)

    model_stem = weights_path.stem.lower()
    is_dual = args.dual_input or ("exp" in model_stem and "baseline" not in model_stem)
    if "exp07" in model_stem:
        args.pseudo_ir_method = "clahe"

    print("=" * 65)
    print("🚗 DEEPSORT RE-ID VEHICLE TRACKING PIPELINE")
    print("=" * 65)
    print(f"• Input Source       : {args.source}")
    print(f"• Model Weights      : {args.weights}")
    print(f"• Dual-Input Mode    : {'Ya (RGB + Pseudo-IR)' if is_dual else 'Tidak (Single RGB)'}")
    if is_dual:
        print(f"• Pseudo-IR Method   : {args.pseudo_ir_method}")
    print(f"• Max Cosine Dist    : {args.max_cosine_dist}")
    print(f"• Max Age            : {args.max_age} frames")
    print(f"• N-Init Confirm     : {args.n_init} frames")
    print(f"• Counting Line Y    : {args.line_y * 100:.1f}% dari tinggi frame")
    print(f"• Output Video       : {args.output if not args.no_save else 'Disabled'}")
    print("=" * 65)

    print("\n⏳ Memuat model YOLO & Re-ID Extractor...")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    print("✓ Model berhasil dimuat!")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka video: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = DeepSORTTracker(
        max_cosine_dist=args.max_cosine_dist,
        max_age=args.max_age,
        n_init=args.n_init,
        trajectory_len=30
    )
    counting_line = CountingLine(line_y_ratio=args.line_y, direction="both")

    writer = None
    if not args.no_save:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_input, (width, height))
        print(f"• Menyiapkan output  : {output_path}")

    print("\n🚀 Memulai proses DeepSORT... (Tekan 'q' untuk berhenti, 'space' untuk jeda)")

    frame_idx = 0
    start_time_all = time.time()
    prev_time = time.time()
    fps_smooth = fps_input
    paused = False
    consecutive_empty = 0
    max_consecutive_empty = 60

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_empty += 1
                    if consecutive_empty > max_consecutive_empty or (total_frames > 0 and frame_idx >= total_frames):
                        break
                    frame_idx += 1
                    continue

                consecutive_empty = 0
                frame_idx += 1

                # 1. Inferensi Model
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

                # 2. Ekstraksi Bounding Box
                detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = model.names.get(cls_id, str(cls_id))
                        detections.append([x1, y1, x2, y2, conf, cls_name])

                # 3. Update DeepSORT
                tracked_objects = tracker.update(frame, detections)

                # 4. Update Counting Line (Split-Lane Aware) dengan pencatatan frame & timestamp
                count_data = counting_line.update(tracked_objects, height, width, frame_idx=frame_idx, fps=fps)

                # 5. Hitung FPS aktual
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps_inst = 1.0 / dt
                    fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst

                # 6. Gambar Visualisasi
                display_frame = frame.copy()
                draw_counting_line(display_frame, count_data)
                draw_tracks(display_frame, tracked_objects, counted_info=count_data.get('counted_directions', {}))
                draw_hud(
                    display_frame,
                    fps=fps_smooth,
                    active_count=len(tracked_objects),
                    count_data=count_data,
                    model_name=weights_path.stem
                )

                # 7. Tulis ke file video
                if writer is not None:
                    writer.write(display_frame)

                # 8. Print progress
                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    print(
                        f"\r[Frame {frame_idx:5d}/{total_frames if total_frames > 0 else '?'}] "
                        f"({pct:5.1f}%) | "
                        f"FPS: {fps_smooth:4.1f} | "
                        f"Active: {len(tracked_objects):2d} | "
                        f"Line Count: {count_data['total']:3d} | "
                        f"Total Unique: {tracker.get_total_count():3d}",
                        end="", flush=True
                    )

            if args.show:
                cv2.imshow("DeepSORT Vehicle Tracking (DEYOLO)", display_frame)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[INFO] Dihentikan oleh pengguna.")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"\n[INFO] {'PAUSED' if paused else 'RESUMED'}")

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan dengan KeyboardInterrupt.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    total_elapsed = time.time() - start_time_all
    avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0

    print("\n\n" + "=" * 65)
    print("🎉 PROSES DEEPSORT SELESAI")
    print("=" * 65)
    print(f"• Total Frame Diproses : {frame_idx} frame")
    print(f"• Total Waktu Eksekusi : {total_elapsed:.2f} detik")
    print(f"• Rata-rata FPS        : {avg_fps:.2f} FPS")
    print(f"• Total Garis Hitung   : {count_data.get('total', 0)} kendaraan (In: {count_data.get('total_in', 0)}, Out: {count_data.get('total_out', 0)})")
    print(f"• Total Unique Track ID: {tracker.get_total_count()} ID")
    print("• Rincian Kelas (Garis Hitung):")
    for cls_name, count in count_data.get('class_counts', {}).items():
        print(f"    - {cls_name:15s}: {count} unit")

    if not args.no_save:
        output_p = Path(args.output).resolve()
        csv_p = output_p.with_name(f"{output_p.stem}_log.csv")
        csv_sum_p = output_p.with_name(f"{output_p.stem}_summary.csv")
        counting_line.export_csv(str(csv_p))
        counting_line.export_summary_csv(str(csv_sum_p))
        print(f"• Video Tersimpan Di   : {output_p}")
        print(f"• Log CSV Tersimpan Di : {csv_p}")
        print(f"• Ringkasan CSV Di     : {csv_sum_p}")
    print("=" * 65)


if __name__ == "__main__":
    run_deepsort()
