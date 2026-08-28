"""
DeepSORT: Deep Simple Online and Realtime Tracking
===================================================
Implementasi mandiri DeepSORT (Wojke et al., 2017):
- Matching Cascade (Penanganan Oklusi Jangka Panjang)
- Mahalanobis Distance Gating + Cosine Distance Metric
- Re-ID Deep Appearance Gallery
- Virtual Counting Line (Tripwire) untuk traffic counting 0% duplikasi
"""

from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter, CHI2_THRESHOLD
from .feature_extractor import FeatureExtractor


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    _count = 0

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None, class_name="vehicle", score=1.0, trajectory_len=30):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = deque(maxlen=100)
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.class_name = str(class_name)
        self.score = float(score)

        self.trajectory = deque(maxlen=trajectory_len)
        self.trajectory.append(self.centroid)

    @property
    def tlwh(self):
        """Bounding box format [x, y, w, h]."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Bounding box format [x1, y1, x2, y2]."""
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.tlbr
        return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

    def predict(self, kf):
        """Prediksi posisi dengan Kalman Filter."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection_xyah, feature, class_name, score):
        """Update state track dengan deteksi baru."""
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection_xyah)
        if feature is not None:
            self.features.append(feature)

        self.hits += 1
        self.time_since_update = 0
        self.class_name = class_name
        self.score = score
        self.trajectory.append(self.centroid)

        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Tandai track jika tidak terdeteksi pada frame ini."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted


def compute_iou_matrix(atlbrs, btlbrs):
    """Menghitung matriks IoU Distance (1 - IoU)."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    for i, a in enumerate(atlbrs):
        for j, b in enumerate(btlbrs):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            inter = w * h
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter
            ious[i, j] = inter / union if union > 0 else 0.0

    return 1.0 - ious


def compute_cosine_distance(track_features_list, det_features):
    """
    Menghitung matriks Cosine Distance: 1 - max(dot(f_track, f_det))
    karena semua fitur telah dinormalisasi L2 (||f||=1).
    """
    num_tracks = len(track_features_list)
    num_dets = len(det_features)
    cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

    for i, tr_feats in enumerate(track_features_list):
        if len(tr_feats) == 0 or num_dets == 0:
            cost_matrix[i, :] = 1.0
            continue
        tr_matrix = np.array(tr_feats)  # shape (num_samples, dim)
        # Dot product: (num_samples, dim) x (dim, num_dets) -> (num_samples, num_dets)
        cosine_sim = np.dot(tr_matrix, det_features.T)
        # Ambil kemiripan maksimum dari seluruh galeri masa lalu objek tersebut
        max_sim = np.max(cosine_sim, axis=0)
        cost_matrix[i, :] = 1.0 - max_sim

    return cost_matrix


def min_cost_matching(cost_matrix, max_distance):
    """Pencocokan biaya minimum dengan Hungarian Algorithm."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_a, unmatched_b = [], [], []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > max_distance:
            unmatched_a.append(r)
            unmatched_b.append(c)
        else:
            matches.append((r, c))

    unmatched_a += [r for r in range(cost_matrix.shape[0]) if r not in row_ind]
    unmatched_b += [c for c in range(cost_matrix.shape[1]) if c not in col_ind]

    return matches, unmatched_a, unmatched_b


class DeepSORTTracker:
    def __init__(self, max_cosine_dist=0.35, max_age=30, n_init=3, trajectory_len=30):
        """
        Parameters:
        -----------
        max_cosine_dist : float
            Threshold jarak kosinus penampilan visual Re-ID (default: 0.35).
        max_age : int
            Batas frame track disimpan saat hilang sebelum dihapus (default: 30).
        n_init : int
            Jumlah deteksi berturut-turut untuk mengonfirmasi status track aktif (default: 3).
        trajectory_len : int
            Panjang riwayat titik centroid yang disimpan.
        """
        self.max_cosine_dist = max_cosine_dist
        self.max_age = max_age
        self.n_init = n_init
        self.trajectory_len = trajectory_len

        self.kf = KalmanFilter()
        self.feature_extractor = FeatureExtractor(feature_dim=128)

        self.tracks = []
        self._next_id = 1

        # Counter statistik
        self.total_unique_count = 0
        self.registered_ids = set()
        self.class_counts = {}

    def update(self, frame, detections):
        """
        Memperbarui DeepSORT dengan frame dan hasil deteksi YOLO.

        Parameters:
        -----------
        frame : np.ndarray
            Citra frame BGR dari OpenCV.
        detections : list
            Format: [[x1, y1, x2, y2, score, class_name], ...]

        Returns:
        --------
        list of dict:
            Track aktif terkonfirmasi:
            [{'id': int, 'bbox': [x1, y1, x2, y2], 'centroid': (cx, cy), 'class_name': str, 'conf': float, 'trajectory': list}, ...]
        """
        # 1. Parse deteksi & convert ke format xyah
        det_bboxes = []
        det_scores = []
        det_classes = []
        det_xyahs = []

        for det in detections:
            if isinstance(det, dict):
                bbox = det.get('bbox', [0, 0, 0, 0])
                score = float(det.get('conf', 1.0))
                cls_name = det.get('class_name', 'vehicle')
            elif len(det) >= 6:
                bbox = det[:4]
                score = float(det[4])
                cls_name = str(det[5])
            elif len(det) == 5:
                bbox = det[:4]
                score = float(det[4])
                cls_name = 'vehicle'
            else:
                continue

            x1, y1, x2, y2 = bbox
            w = max(1.0, float(x2 - x1))
            h = max(1.0, float(y2 - y1))
            cx = float(x1) + w / 2.0
            cy = float(y1) + h / 2.0
            aspect = w / h

            det_bboxes.append([x1, y1, x2, y2])
            det_scores.append(score)
            det_classes.append(cls_name)
            det_xyahs.append(np.array([cx, cy, aspect, h]))

        det_bboxes = np.array(det_bboxes) if len(det_bboxes) > 0 else np.empty((0, 4))
        det_xyahs = np.array(det_xyahs) if len(det_xyahs) > 0 else np.empty((0, 4))

        # 2. Ekstraksi visual appearance features (Re-ID CNN)
        features = self.feature_extractor.extract(frame, det_bboxes)

        # 3. Prediksi posisi semua track dengan Kalman Filter
        for track in self.tracks:
            track.predict(self.kf)

        # 4. Lakukan Matching Cascade (Deep Appearance + Mahalanobis Gating)
        matches, unmatched_tracks, unmatched_detections = self._match(det_xyahs, features)

        # 5. Update track yang berhasil matched
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                self.kf,
                det_xyahs[det_idx],
                features[det_idx],
                det_classes[det_idx],
                det_scores[det_idx]
            )

        # 6. Tandai track yang missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 7. Inisialisasi track baru untuk deteksi yang belum terpasang
        for det_idx in unmatched_detections:
            self._initiate_track(
                det_xyahs[det_idx],
                features[det_idx],
                det_classes[det_idx],
                det_scores[det_idx]
            )

        # 8. Hapus track yang berstatus Deleted
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # 9. Update counter statistik unik
        for track in self.tracks:
            if track.is_confirmed() and track.track_id not in self.registered_ids:
                self.registered_ids.add(track.track_id)
                self.total_unique_count += 1
                cls_k = str(track.class_name)
                self.class_counts[cls_k] = self.class_counts.get(cls_k, 0) + 1

        # 10. Format hasil akhir (hanya return track yang terkonfirmasi)
        active_tracks = []
        for track in self.tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                x1, y1, x2, y2 = [int(v) for v in track.tlbr]
                active_tracks.append({
                    'id': track.track_id,
                    'bbox': [x1, y1, x2, y2],
                    'centroid': track.centroid,
                    'class_name': track.class_name,
                    'conf': track.score,
                    'trajectory': list(track.trajectory)
                })

        return active_tracks

    def _match(self, det_xyahs, features):
        """
        Algoritma Matching Cascade DeepSORT:
        1. Cascade Matching untuk Confirmed Tracks (Appearance Cosine + Mahalanobis)
        2. IoU Matching Fallback untuk Unmatched & Tentative Tracks
        """
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # =====================================================================
        # BAGIAN 1: MATCHING CASCADE (Berdasarkan Umur / Age Track)
        # =====================================================================
        matches = []
        unmatched_detections = list(range(len(features)))

        # Iterasi dari umur 1 s.d. max_age
        for age in range(1, self.max_age + 1):
            if len(unmatched_detections) == 0:
                break

            # Ambil track yang waktu tidak terdeteksinya == age
            track_indices = [k for k in confirmed_tracks if self.tracks[k].time_since_update == age]
            if len(track_indices) == 0:
                continue

            sub_features = [self.tracks[k].features for k in track_indices]
            sub_dets = features[unmatched_detections]

            # 1. Hitung Cosine Distance Matrix
            cost_matrix = compute_cosine_distance(sub_features, sub_dets)

            # 2. Mahalanobis Gating: Batalkan jika posisi fisik tidak masuk akal (Chi2 > 9.4877)
            if len(det_xyahs) > 0:
                for row, tr_idx in enumerate(track_indices):
                    track = self.tracks[tr_idx]
                    maha_dists = self.kf.gating_distance(track.mean, track.covariance, det_xyahs[unmatched_detections])
                    cost_matrix[row, maha_dists > CHI2_THRESHOLD] = 1e5

            # 3. Hungarian Matching
            sub_matches, sub_u_tracks, sub_u_dets = min_cost_matching(cost_matrix, max_distance=self.max_cosine_dist)

            for r, c in sub_matches:
                matches.append((track_indices[r], unmatched_detections[c]))

            unmatched_detections = [unmatched_detections[c] for c in sub_u_dets]

        matched_tracks = [m[0] for m in matches]
        unmatched_confirmed = [k for k in confirmed_tracks if k not in matched_tracks]

        # =====================================================================
        # BAGIAN 2: IOU MATCHING FALLBACK
        # =====================================================================
        # Gabungkan track confirmed yang baru saja terlewat (time_since_update == 1) dan unconfirmed tracks
        iou_track_candidates = [k for k in unmatched_confirmed if self.tracks[k].time_since_update == 1] + unconfirmed_tracks
        unmatched_confirmed_rest = [k for k in unmatched_confirmed if self.tracks[k].time_since_update > 1]

        if len(iou_track_candidates) > 0 and len(unmatched_detections) > 0:
            track_tlbrs = [self.tracks[k].tlbr for k in iou_track_candidates]
            det_tlbrs = []
            for det_idx in unmatched_detections:
                cx, cy, a, h = det_xyahs[det_idx]
                w = a * h
                det_tlbrs.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

            iou_cost = compute_iou_matrix(track_tlbrs, det_tlbrs)
            iou_matches, iou_u_tracks, iou_u_dets = min_cost_matching(iou_cost, max_distance=0.70)

            for r, c in iou_matches:
                matches.append((iou_track_candidates[r], unmatched_detections[c]))

            unmatched_tracks = [iou_track_candidates[r] for r in iou_u_tracks] + unmatched_confirmed_rest
            unmatched_detections = [unmatched_detections[c] for c in iou_u_dets]
        else:
            unmatched_tracks = iou_track_candidates + unmatched_confirmed_rest

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection_xyah, feature, class_name, score):
        """Membuat instance track baru."""
        mean, covariance = self.kf.initiate(detection_xyah)
        self.tracks.append(Track(
            mean=mean,
            covariance=covariance,
            track_id=self._next_id,
            n_init=self.n_init,
            max_age=self.max_age,
            feature=feature,
            class_name=class_name,
            score=score,
            trajectory_len=self.trajectory_len
        ))
        self._next_id += 1

    def get_total_count(self):
        return self.total_unique_count

    def get_class_counts(self):
        return dict(self.class_counts)

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
        self.total_unique_count = 0
        self.registered_ids.clear()
        self.class_counts.clear()


# =====================================================================
# VIRTUAL COUNTING LINE (TRIPWIRE) & SPLIT-LANE COUNTING
# =====================================================================
class CountingLine:
    def __init__(self, line_y_ratio=0.55, line_y_left_ratio=0.70, line_y_right_ratio=0.50,
                 split_x_ratio=0.50, mode="split", direction="both", buffer_px=15):
        """
        Garis Hitung Virtual (Tripwire) dengan dukungan Split-Lane (Lajur Kiri vs Kanan).

        Parameters:
        -----------
        line_y_ratio : float
            Posisi garis horizontal tunggal (mode single line, default: 0.55).
        line_y_left_ratio : float
            Posisi garis horizontal untuk Lajur Kiri (Arah OUT / Ke Atas, default: 0.70).
        line_y_right_ratio : float
            Posisi garis horizontal untuk Lajur Kanan (Arah IN / Ke Bawah, default: 0.50).
        split_x_ratio : float
            Pembagi batas horizontal antara Lajur Kiri dan Lajur Kanan (default: 0.50).
        mode : str
            'split' (Split-Lane per lajur) atau 'single' (Garis tunggal penuh).
        direction : str
            'both', 'down', atau 'up'.
        """
        self.line_y_ratio = line_y_ratio
        self.line_y_left_ratio = line_y_left_ratio
        self.line_y_right_ratio = line_y_right_ratio
        self.split_x_ratio = split_x_ratio
        self.mode = mode
        self.direction = direction
        self.buffer_px = buffer_px

        self.total_in = 0
        self.total_out = 0
        self.counted_ids = set()
        self.counted_directions = {}  # id -> 'IN' / 'OUT'
        self.class_counts = {}

    def update(self, tracked_objects, frame_height, frame_width=None):
        """
        Evaluasi apakah objek melintasi garis hitung pada frame ini.
        Mendukung Split-Lane (Lajur Kiri OUT vs Lajur Kanan IN) serta Bbox-Span.
        """
        if frame_width is None:
            frame_width = int(frame_height * (16 / 9))

        split_x = int(frame_width * self.split_x_ratio)
        line_y_left = int(frame_height * self.line_y_left_ratio)
        line_y_right = int(frame_height * self.line_y_right_ratio)
        line_y_single = int(frame_height * self.line_y_ratio)

        for obj in tracked_objects:
            obj_id = obj['id']
            trajectory = obj['trajectory']
            cls_name = obj['class_name']
            bbox = obj.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            cx, cy = obj['centroid']

            if obj_id in self.counted_ids:
                continue

            if len(trajectory) >= 2:
                prev_cx, prev_cy = trajectory[-2]
                curr_cx, curr_cy = trajectory[-1]
                dy = curr_cy - prev_cy  # dy < 0: moving UP (OUT), dy > 0: moving DOWN (IN)

                if self.mode == "split":
                    is_left_lane = curr_cx < split_x

                    if is_left_lane:
                        # ==================== LAJUR KIRI: ARAH OUT (KE ATAS) ====================
                        target_y = line_y_left
                        crossed_up = (prev_cy > target_y >= curr_cy)
                        bbox_crossed_up = (dy < 0 and y1 <= target_y <= y2 and prev_cy >= target_y - 20)

                        if (crossed_up or bbox_crossed_up) and self.direction in ["up", "both"]:
                            self.total_out += 1
                            self.counted_ids.add(obj_id)
                            self.counted_directions[obj_id] = "OUT"
                            cls_k = str(cls_name)
                            self.class_counts[cls_k] = self.class_counts.get(cls_k, 0) + 1

                    else:
                        # ==================== LAJUR KANAN: ARAH IN (KE BAWAH) ====================
                        target_y = line_y_right
                        crossed_down = (prev_cy < target_y <= curr_cy)
                        bbox_crossed_down = (dy > 0 and y1 <= target_y <= y2 and prev_cy <= target_y + 20)

                        if (crossed_down or bbox_crossed_down) and self.direction in ["down", "both"]:
                            self.total_in += 1
                            self.counted_ids.add(obj_id)
                            self.counted_directions[obj_id] = "IN"
                            cls_k = str(cls_name)
                            self.class_counts[cls_k] = self.class_counts.get(cls_k, 0) + 1

                else:
                    # ==================== SINGLE LINE MODE ====================
                    target_y = line_y_single
                    # Melintas ke bawah (Down / In)
                    if prev_cy < target_y <= curr_cy or (dy > 0 and y1 <= target_y <= y2 and prev_cy <= target_y + 15):
                        if self.direction in ["down", "both"]:
                            self.total_in += 1
                            self.counted_ids.add(obj_id)
                            self.counted_directions[obj_id] = "IN"
                            cls_k = str(cls_name)
                            self.class_counts[cls_k] = self.class_counts.get(cls_k, 0) + 1

                    # Melintas ke atas (Up / Out)
                    elif prev_cy > target_y >= curr_cy or (dy < 0 and y1 <= target_y <= y2 and prev_cy >= target_y - 15):
                        if self.direction in ["up", "both"]:
                            self.total_out += 1
                            self.counted_ids.add(obj_id)
                            self.counted_directions[obj_id] = "OUT"
                            cls_k = str(cls_name)
                            self.class_counts[cls_k] = self.class_counts.get(cls_k, 0) + 1

        return {
            'total_in': self.total_in,
            'total_out': self.total_out,
            'total': self.total_in + self.total_out,
            'class_counts': dict(self.class_counts),
            'line_y': line_y_single,
            'line_y_left': line_y_left,
            'line_y_right': line_y_right,
            'split_x': split_x,
            'mode': self.mode,
            'counted_ids': set(self.counted_ids),
            'counted_directions': dict(self.counted_directions)
        }

    def reset(self):
        self.total_in = 0
        self.total_out = 0
        self.counted_ids.clear()
        self.counted_directions.clear()
        self.class_counts.clear()
