"""
Script Testing/Prediction untuk Baseline YOLOv8n dan DEYOLO
==============================================================
Sesuaikan path di bagian KONFIGURASI sebelum menjalankan.
"""

import cv2
import numpy as np
from pathlib import Path

# ========================= KONFIGURASI =========================
RGB_IMAGE_PATH = "sample_rgb.jpg"          # gambar input RGB

BASELINE_WEIGHTS = "/kaggle/working/baseline/weights/best.pt"
DEYOLO_WEIGHTS   = "/kaggle/working/exp05/weights/best.pt"   # ganti sesuai eksperimen yg mau ditest

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
OUTPUT_DIR = "predict_results"
# =================================================================

Path(OUTPUT_DIR).mkdir(exist_ok=True)


def generate_pseudo_ir(rgb_path, method="sobel"):
    """Generate Pseudo-IR dari gambar RGB, method: 'sobel' atau 'clahe'."""
    img = cv2.imread(rgb_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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


# ===================================================================
# BAGIAN 1 — BASELINE YOLOv8n (single input, langsung pakai Ultralytics API standar)
# ===================================================================
def predict_baseline():
    from ultralytics import YOLO

    model = YOLO(BASELINE_WEIGHTS)
    results = model.predict(
        source=RGB_IMAGE_PATH,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,
        project=OUTPUT_DIR,
        name="baseline",
        exist_ok=True,
    )

    # Print ringkasan deteksi
    for r in results:
        print(f"\n[BASELINE] Detections: {len(r.boxes)}")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            print(f"  - {cls_name}: confidence {conf:.3f}")

    return results


# ===================================================================
# BAGIAN 2 — DEYOLO (dual input: RGB + Pseudo-IR)
# ===================================================================
def predict_deyolo(pseudo_ir_method="sobel"):
    """
    PENTING: Bagian ini perlu disesuaikan dengan API dari codebase/fork DEYOLO
    yang kamu pakai untuk training. Ultralytics YOLO() standar tidak mendukung
    dual-input secara native, jadi kemungkinan besar DEYOLO fork-mu punya
    method/class predict khusus.

    Ada 2 pola umum yang biasa dipakai fork DEYOLO (cek repo yang kamu pakai,
    biasanya di file predict.py atau val.py bawaan repo):

    POLA A — model menerima folder dengan subfolder images/ dan images_ir/
    POLA B — model.predict() menerima list [rgb_path, ir_path]
    """
    # 1. Generate Pseudo-IR dulu dari RGB
    pseudo_ir_img = generate_pseudo_ir(RGB_IMAGE_PATH, method=pseudo_ir_method)
    pseudo_ir_path = f"{OUTPUT_DIR}/temp_pseudo_ir.jpg"
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    cv2.imwrite(pseudo_ir_path, pseudo_ir_img)
    print(f"Pseudo-IR ({pseudo_ir_method}) generated: {pseudo_ir_path}")

    # 2. Load model DEYOLO
    from ultralytics import YOLO
    model = YOLO(DEYOLO_WEIGHTS)

    # 3. Predict — format resmi repo chips96/DEYOLO: list of [rgb_path, ir_path]
    #    Referensi: https://github.com/chips96/DEYOLO#predict
    results = model.predict(
        source=[[RGB_IMAGE_PATH, pseudo_ir_path]],  # list berisi 1 pasang [rgb, ir]
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,
        project=OUTPUT_DIR,
        name="deyolo",
        exist_ok=True,
    )

    for r in results:
        print(f"\n[DEYOLO - {pseudo_ir_method}] Detections: {len(r.boxes)}")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            print(f"  - {cls_name}: confidence {conf:.3f}")

    return results


if __name__ == "__main__":
    print("=== Menjalankan prediksi baseline ===")
    predict_baseline()

    print("\n=== Menjalankan prediksi DEYOLO (Sobel) ===")
    predict_deyolo(pseudo_ir_method="sobel")

    print("\nSelesai. Hasil gambar dengan bounding box tersimpan di folder:", OUTPUT_DIR)
