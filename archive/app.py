"""
DEYOLO Vehicle Detection — Testing App
========================================
Cara jalankan:
    streamlit run app.py

Sebelum jalan, sesuaikan MODEL_REGISTRY di bawah dengan path best.pt
tiap eksperimenmu.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile

# ========================= KONFIGURASI =========================
# Sesuaikan path ini dengan lokasi weights hasil training-mu
MODEL_REGISTRY = {
    "EXP-01 — Baseline (YOLOv8n)": {
        "path": "weights/baseline_best.pt",
        "dual_input": False,
        "pseudo_ir_method": None,
    },
    "EXP-02 — DEYOLOn-Default (3x7, r16, Sobel)": {
        "path": "weights/exp02_best.pt",
        "dual_input": True,
        "pseudo_ir_method": "sobel",
    },
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
}

CONF_THRESHOLD_DEFAULT = 0.25
IOU_THRESHOLD_DEFAULT = 0.7
# =================================================================


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
def load_model(weights_path):
    """Load model YOLO, di-cache biar ga reload tiap kali ganti gambar."""
    from ultralytics import YOLO
    return YOLO(weights_path)


def run_inference(model, rgb_path, conf, iou, dual_input, pseudo_ir_path=None):
    """
    Jalankan prediksi. Bagian dual_input (DEYOLO) mungkin perlu disesuaikan
    dengan API custom dari fork DEYOLO yang kamu pakai — lihat catatan
    di bawah fungsi ini.
    """
    if not dual_input:
        results = model.predict(source=rgb_path, conf=conf, iou=iou, save=False)
    else:
        # === SESUAIKAN BAGIAN INI DENGAN API DEYOLO FORK-MU ===
        # Placeholder: masih coba source RGB tunggal dulu.
        # Kalau fork DEYOLO-mu butuh source=[rgb_path, ir_path] atau
        # source folder dengan struktur khusus, ganti baris di bawah.
        try:
            results = model.predict(source=rgb_path, conf=conf, iou=iou, save=False)
        except TypeError:
            results = model.predict(source=[rgb_path, pseudo_ir_path], conf=conf, iou=iou, save=False)

    return results


# ========================= UI =========================
st.set_page_config(page_title="DEYOLO Vehicle Detection", layout="wide")
st.title("🚗 DEYOLO Vehicle Detection — Rain Condition Testing")
st.caption("Pilih model, upload gambar, dan jalankan deteksi kendaraan.")

col_control, col_result = st.columns([1, 2])

with col_control:
    st.subheader("1. Pilih Model")
    selected_model_name = st.selectbox("Konfigurasi model:", list(MODEL_REGISTRY.keys()))
    model_config = MODEL_REGISTRY[selected_model_name]

    st.subheader("2. Upload Gambar")
    uploaded_file = st.file_uploader("Pilih gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

    st.subheader("3. Pengaturan Threshold")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, CONF_THRESHOLD_DEFAULT, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.0, 1.0, IOU_THRESHOLD_DEFAULT, 0.05)

    run_button = st.button("🔍 Jalankan Deteksi", type="primary", use_container_width=True)

with col_result:
    if uploaded_file is not None:
        # Simpan file upload ke temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            rgb_path = tmp.name

        rgb_img = cv2.imread(rgb_path)

        if not run_button:
            st.subheader("Preview Gambar")
            st.image(Image.open(uploaded_file), caption="Gambar Input (RGB)", use_container_width=True)

        if run_button:
            with st.spinner("Menjalankan model..."):
                pseudo_ir_path = None
                pseudo_ir_display = None

                if model_config["dual_input"]:
                    pseudo_ir_img = generate_pseudo_ir(rgb_img, method=model_config["pseudo_ir_method"])
                    pseudo_ir_path = rgb_path.replace(".jpg", "_ir.jpg")
                    cv2.imwrite(pseudo_ir_path, pseudo_ir_img)
                    pseudo_ir_display = cv2.cvtColor(pseudo_ir_img, cv2.COLOR_BGR2RGB)

                try:
                    model = load_model(model_config["path"])
                    results = run_inference(
                        model, rgb_path, conf_threshold, iou_threshold,
                        model_config["dual_input"], pseudo_ir_path
                    )

                    result_img = results[0].plot()  # gambar dengan bounding box
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                    # Tampilkan gambar
                    if model_config["dual_input"]:
                        c1, c2, c3 = st.columns(3)
                        c1.image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), caption="RGB (Original)", use_container_width=True)
                        c2.image(pseudo_ir_display, caption=f"Pseudo-IR ({model_config['pseudo_ir_method']})", use_container_width=True)
                        c3.image(result_img_rgb, caption="Hasil Deteksi", use_container_width=True)
                    else:
                        st.image(result_img_rgb, caption="Hasil Deteksi", use_container_width=True)

                    # Tabel hasil deteksi
                    st.subheader("Detail Deteksi")
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        detection_data = []
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            cls_name = model.names[cls_id]
                            detection_data.append({"Class": cls_name, "Confidence": f"{conf:.3f}"})
                        st.table(detection_data)
                        st.success(f"Total {len(boxes)} objek terdeteksi.")
                    else:
                        st.warning("Tidak ada objek terdeteksi pada threshold ini.")

                except FileNotFoundError:
                    st.error(
                        f"File model tidak ditemukan: `{model_config['path']}`\n\n"
                        "Sesuaikan path di MODEL_REGISTRY pada bagian atas kode."
                    )
                except Exception as e:
                    st.error(f"Terjadi error saat menjalankan model: {e}")
    else:
        st.info("👈 Upload gambar dulu di panel sebelah kiri untuk mulai.")
