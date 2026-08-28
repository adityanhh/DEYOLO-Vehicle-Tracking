"""
Euclidean Distance Tracker (Centroid Tracker)
==============================================
Modul pelacakan objek berbasis jarak Euclidean untuk aplikasi deteksi kendaraan.

Cara kerja:
1. Menerima bounding box deteksi [x1, y1, x2, y2, score, class_id/name]
2. Menghitung centroid (titik tengah) dari setiap bounding box
3. Menghitung matriks jarak Euclidean antara centroid baru dan objek yang sudah terdaftar
4. Melakukan matching berpasangan dengan jarak minimum <= max_distance
5. Mendaftarkan objek baru, memperbarui objek cocok, dan menghapus objek yang hilang > max_disappeared frames
6. Menyimpan riwayat lintasan (trajectory) untuk visualisasi pergerakan
"""

import math
from collections import deque
import numpy as np


class EuclideanDistTracker:
    def __init__(self, max_distance=65, max_disappeared=20, trajectory_len=30):
        """
        Inisialisasi Euclidean Distance Tracker.

        Parameters:
        -----------
        max_distance : int or float
            Batas jarak maksimum (dalam pixel) antar centroid di dua frame berurutan
            untuk dianggap sebagai objek yang sama.
        max_disappeared : int
            Jumlah frame berturut-turut sebuah objek boleh tidak terdeteksi
            sebelum ID-nya dihapus dari daftar pelacakan.
        trajectory_len : int
            Jumlah titik centroid masa lalu yang disimpan untuk menggambar jejak (trail).
        """
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.trajectory_len = trajectory_len

        # Counter untuk ID unik
        self.next_object_id = 1

        # Data pelacakan aktif:
        # id -> (cx, cy)
        self.centroids = {}
        # id -> [x1, y1, x2, y2]
        self.bboxes = {}
        # id -> class_name / class_id
        self.class_names = {}
        # id -> confidence score
        self.confidences = {}
        # id -> jumlah frame menghilang berturut-turut
        self.disappeared = {}
        # id -> deque([(cx, cy), ...]) riwayat posisi
        self.trajectories = {}

        # Statistik akumulasi unik
        self.total_unique_count = 0
        self.class_counts = {}

    def register(self, centroid, bbox, class_name, confidence):
        """Mendaftarkan objek baru ke dalam tracker."""
        obj_id = self.next_object_id
        self.centroids[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.class_names[obj_id] = class_name
        self.confidences[obj_id] = confidence
        self.disappeared[obj_id] = 0

        self.trajectories[obj_id] = deque(maxlen=self.trajectory_len)
        self.trajectories[obj_id].append(centroid)

        # Update counter global
        self.total_unique_count += 1
        cls_key = str(class_name)
        self.class_counts[cls_key] = self.class_counts.get(cls_key, 0) + 1

        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id):
        """Menghapus objek dari daftar pelacakan aktif."""
        self.centroids.pop(obj_id, None)
        self.bboxes.pop(obj_id, None)
        self.class_names.pop(obj_id, None)
        self.confidences.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)
        self.trajectories.pop(obj_id, None)

    def update(self, detections):
        """
        Memperbarui status pelacak dengan hasil deteksi frame saat ini.

        Parameters:
        -----------
        detections : list of dict atau list of tuple/list
            Format yang didukung:
            1. List of dict: [{'bbox': [x1, y1, x2, y2], 'class_name': str, 'conf': float}, ...]
            2. List of list/tuple: [[x1, y1, x2, y2, conf, class_name], ...] atau [[x1, y1, x2, y2], ...]

        Returns:
        --------
        list of dict:
            Objek-objek yang sedang aktif terlacak pada frame ini:
            [
                {
                    'id': int,
                    'bbox': [x1, y1, x2, y2],
                    'centroid': (cx, cy),
                    'class_name': str,
                    'conf': float,
                    'trajectory': [(cx, cy), ...]
                },
                ...
            ]
        """
        # Standarisasi input detections
        input_bboxes = []
        input_classes = []
        input_confs = []
        input_centroids = []

        for det in detections:
            if isinstance(det, dict):
                bbox = det.get('bbox', [0, 0, 0, 0])
                cls_name = det.get('class_name', 'vehicle')
                conf = det.get('conf', 1.0)
            elif len(det) >= 6:
                bbox = det[:4]
                conf = float(det[4])
                cls_name = str(det[5])
            elif len(det) == 5:
                bbox = det[:4]
                conf = float(det[4])
                cls_name = 'vehicle'
            else:
                bbox = det[:4]
                conf = 1.0
                cls_name = 'vehicle'

            x1, y1, x2, y2 = [int(v) for v in bbox]
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)

            input_bboxes.append([x1, y1, x2, y2])
            input_classes.append(cls_name)
            input_confs.append(conf)
            input_centroids.append((cx, cy))

        # KASUS 1: Jika tidak ada deteksi baru pada frame ini
        if len(input_centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self._get_active_tracks()

        # KASUS 2: Jika belum ada objek yang sedang dilacak sebelumnya
        if len(self.centroids) == 0:
            for i in range(len(input_centroids)):
                self.register(
                    input_centroids[i],
                    input_bboxes[i],
                    input_classes[i],
                    input_confs[i]
                )
            return self._get_active_tracks()

        # KASUS 3: Ada objek lama dan ada deteksi baru -> Hitung Jarak Euclidean
        object_ids = list(self.centroids.keys())
        existing_centroids = [self.centroids[oid] for oid in object_ids]

        # Buat Distance Matrix: Shape (num_existing_objects, num_detections)
        # Baris i = ID lama ke-i, Kolom j = Deteksi baru ke-j
        num_existing = len(existing_centroids)
        num_new = len(input_centroids)
        dist_matrix = np.zeros((num_existing, num_new), dtype=np.float32)

        for i in range(num_existing):
            for j in range(num_new):
                ex, ey = existing_centroids[i]
                nx, ny = input_centroids[j]
                dist = math.sqrt((nx - ex) ** 2 + (ny - ey) ** 2)
                dist_matrix[i, j] = dist

        # Pencocokan greedy berdasarkan jarak terkecil
        # Cari urutan jarak terpendek di seluruh matriks
        matched_rows = set()
        matched_cols = set()

        # Sort semua pasangan (i, j) berdasarkan nilai dist terkecil
        row_indices = dist_matrix.min(axis=1).argsort()

        for r in row_indices:
            # Ambil kolom dengan jarak terpendek untuk baris ini
            c = dist_matrix[r].argmin()

            if r in matched_rows or c in matched_cols:
                continue

            # Periksa apakah jarak <= max_distance
            if dist_matrix[r, c] <= self.max_distance:
                obj_id = object_ids[r]
                # Update data objek yang cocok
                self.centroids[obj_id] = input_centroids[c]
                self.bboxes[obj_id] = input_bboxes[c]
                self.class_names[obj_id] = input_classes[c]
                self.confidences[obj_id] = input_confs[c]
                self.disappeared[obj_id] = 0
                self.trajectories[obj_id].append(input_centroids[c])

                matched_rows.add(r)
                matched_cols.add(c)

        # Deteksi baris (ID lama) yang tidak dapat pasangan -> increment disappeared
        unmatched_rows = set(range(num_existing)) - matched_rows
        for r in unmatched_rows:
            obj_id = object_ids[r]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        # Deteksi kolom (deteksi baru) yang tidak dapat pasangan -> register ID baru
        unmatched_cols = set(range(num_new)) - matched_cols
        for c in unmatched_cols:
            self.register(
                input_centroids[c],
                input_bboxes[c],
                input_classes[c],
                input_confs[c]
            )

        return self._get_active_tracks()

    def _get_active_tracks(self):
        """Mengembalikan list objek yang aktif terdeteksi/terlacak."""
        active_tracks = []
        for obj_id, centroid in self.centroids.items():
            # Hanya kembalikan objek yang tidak sedang hilang pada frame saat ini
            if self.disappeared.get(obj_id, 0) == 0:
                active_tracks.append({
                    'id': obj_id,
                    'bbox': self.bboxes[obj_id],
                    'centroid': centroid,
                    'class_name': self.class_names.get(obj_id, 'vehicle'),
                    'conf': self.confidences.get(obj_id, 0.0),
                    'trajectory': list(self.trajectories.get(obj_id, []))
                })
        return active_tracks

    def get_total_count(self):
        """Mengembalikan total objek unik yang pernah terdaftar."""
        return self.total_unique_count

    def get_class_counts(self):
        """Mengembalikan dictionary jumlah kendaraan per kelas."""
        return dict(self.class_counts)

    def reset(self):
        """Reset seluruh state tracker."""
        self.next_object_id = 1
        self.centroids.clear()
        self.bboxes.clear()
        self.class_names.clear()
        self.confidences.clear()
        self.disappeared.clear()
        self.trajectories.clear()
        self.total_unique_count = 0
        self.class_counts.clear()
