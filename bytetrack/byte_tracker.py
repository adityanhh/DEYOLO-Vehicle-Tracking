"""
ByteTrack: Multi-Object Tracking (Two-Stage Matching)
======================================================
Implementasi mandiri ByteTrack (Zhang et al., 2022) dengan Kalman Filter 8-State
dan Hungarian Algorithm (linear_sum_assignment) via SciPy.

Dilengkapi dengan modul Virtual Counting Line (Tripwire) untuk traffic counting bebas duplikat.
"""

from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    shared_kalman = KalmanFilter()
    _count = 0

    def __init__(self, tlwh, score, class_name="vehicle", trajectory_len=30):
        # Format bbox: [top_left_x, top_left_y, width, height]
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

        self.score = float(score)
        self.class_name = str(class_name)
        self.tracklet_len = 0
        self.state = TrackState.New

        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

        self.trajectory_len = trajectory_len
        self.trajectory = deque(maxlen=trajectory_len)

    @property
    def tlwh(self):
        """Ambil bounding box format [x, y, w, h]."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Ambil bounding box format [x1, y1, x2, y2]."""
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def centroid(self):
        """Ambil titik tengah (cx, cy)."""
        x1, y1, x2, y2 = self.tlbr
        return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_counter():
        STrack._count = 0

    def activate(self, kalman_filter, frame_id):
        """Aktivasi track baru."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.trajectory.append(self.centroid)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reaktivasi track yang sempat lost."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.class_name = new_track.class_name
        self.trajectory.append(self.centroid)

    def update(self, new_track, frame_id):
        """Update track dengan deteksi baru."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.class_name = new_track.class_name
        self.trajectory.append(self.centroid)

    def predict(self):
        """Prediksi posisi frame berikutnya menggunakan Kalman Filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert [top_left_x, top_left_y, w, h] to [center_x, center_y, aspect_ratio, h]."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


def compute_iou_matrix(atlbrs, btlbrs):
    """Menghitung matriks IoU Distance (1 - IoU) antara dua grup bounding box."""
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


def linear_assignment(cost_matrix, thresh):
    """Pencocokan Hungarian Algorithm dengan batas threshold cost."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_a, unmatched_b = [], [], []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            unmatched_a.append(r)
            unmatched_b.append(c)
        else:
            matches.append((r, c))

    unmatched_a += [r for r in range(cost_matrix.shape[0]) if r not in row_ind]
    unmatched_b += [c for c in range(cost_matrix.shape[1]) if c not in col_ind]

    return np.asarray(matches, dtype=int), np.asarray(unmatched_a, dtype=int), np.asarray(unmatched_b, dtype=int)


class BYTETracker:
    def __init__(self, track_thresh=0.45, match_thresh=0.80, track_buffer=30, frame_rate=30, trajectory_len=30):
        """
        Inisialisasi ByteTrack Tracker.

        Parameters:
        -----------
        track_thresh : float
            Threshold untuk memisahkan High-Score Detections vs Low-Score Detections.
        match_thresh : float
            IoU distance threshold untuk asosiasi tahap 1.
        track_buffer : int
            Jumlah frame maksimum track yang 'Lost' disimpan sebelum dihapus (deregistrasi).
        frame_rate : int
            FPS video.
        trajectory_len : int
            Panjang riwayat titik centroid yang disimpan.
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.trajectory_len = trajectory_len

        self.tracked_stracks = []  # list of STrack
        self.lost_stracks = []     # list of STrack
        self.removed_stracks = []  # list of STrack

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        # Statistik Unik Akumulasi
        self.total_unique_count = 0
        self.registered_ids = set()
        self.class_counts = {}

    def update(self, detections):
        """
        Memperbarui ByteTrack dengan deteksi pada frame saat ini.

        Parameters:
        -----------
        detections : list
            Format: [[x1, y1, x2, y2, score, class_name], ...] atau list of dict.

        Returns:
        --------
        list of dict:
            Track aktif pada frame saat ini:
            [{'id': int, 'bbox': [x1, y1, x2, y2], 'centroid': (cx, cy), 'class_name': str, 'conf': float, 'trajectory': list}, ...]
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. Parse dan Pisahkan Deteksi (High Score vs Low Score)
        dets_high = []
        dets_low = []

        for det in detections:
            if isinstance(det, dict):
                x1, y1, x2, y2 = det.get('bbox', [0, 0, 0, 0])
                score = float(det.get('conf', 1.0))
                cls_name = det.get('class_name', 'vehicle')
            elif len(det) >= 6:
                x1, y1, x2, y2 = det[:4]
                score = float(det[4])
                cls_name = str(det[5])
            elif len(det) == 5:
                x1, y1, x2, y2 = det[:4]
                score = float(det[4])
                cls_name = 'vehicle'
            else:
                continue

            tlwh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            track_item = STrack(tlwh, score, cls_name, self.trajectory_len)

            if score >= self.track_thresh:
                dets_high.append(track_item)
            elif score >= 0.10:  # low score detection threshold
                dets_low.append(track_item)

        # 2. Pisahkan Track Aktif & Lost
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Gabungkan Tracked + Lost untuk dicocokkan di Tahap 1
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Prediksi posisi frame ini untuk seluruh pool track dengan Kalman Filter
        for strack in strack_pool:
            strack.predict()

        # =====================================================================
        # TAHAP 1: Match High-Score Detections dengan Track Pool (IoU Distance)
        # =====================================================================
        dists = compute_iou_matrix([t.tlbr for t in strack_pool], [d.tlbr for d in dets_high])
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = dets_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # =====================================================================
        # TAHAP 2: Match Low-Score Detections dengan Sisa Track Unmatched
        # =====================================================================
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists_low = compute_iou_matrix([t.tlbr for t in r_tracked_stracks], [d.tlbr for d in dets_low])
        matches_low, u_track_low, u_detection_low = linear_assignment(dists_low, thresh=0.50)

        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det = dets_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Track yang tetap tidak cocok -> Pindahkan ke Lost
        for it in u_track_low:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        # =====================================================================
        # TAHAP 3: Match Unconfirmed Tracks dengan Sisa Deteksi High-Score
        # =====================================================================
        dets_high_unmatched = [dets_high[i] for i in u_detection]
        dists_unconf = compute_iou_matrix([t.tlbr for t in unconfirmed], [d.tlbr for d in dets_high_unmatched])
        matches_unconf, u_unconfirmed, u_detection_unconf = linear_assignment(dists_unconf, thresh=0.70)

        for itracked, idet in matches_unconf:
            unconfirmed[itracked].update(dets_high_unmatched[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        # =====================================================================
        # TAHAP 4: Inisialisasi Track Baru dari Sisa Deteksi High-Score
        # =====================================================================
        for inew in u_detection_unconf:
            track = dets_high_unmatched[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # =====================================================================
        # TAHAP 5: Manajemen State & Hapus Track yang Hilang Terlalu Lama
        # =====================================================================
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        # Update daftar track
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # Update counter statistik unik
        for track in self.tracked_stracks:
            if track.is_activated and track.track_id not in self.registered_ids:
                self.registered_ids.add(track.track_id)
                self.total_unique_count += 1
                cls_k = str(track.class_name)
                self.class_counts[cls_k] = self.class_counts.get(cls_k, 0) + 1

        # Format output
        output_stracks = []
        for track in self.tracked_stracks:
            if track.is_activated:
                x1, y1, x2, y2 = [int(v) for v in track.tlbr]
                output_stracks.append({
                    'id': track.track_id,
                    'bbox': [x1, y1, x2, y2],
                    'centroid': track.centroid,
                    'class_name': track.class_name,
                    'conf': track.score,
                    'trajectory': list(track.trajectory)
                })

        return output_stracks

    def get_total_count(self):
        return self.total_unique_count

    def get_class_counts(self):
        return dict(self.class_counts)

    def reset(self):
        STrack.reset_counter()
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        self.total_unique_count = 0
        self.registered_ids.clear()
        self.class_counts.clear()


# =====================================================================
# VIRTUAL COUNTING LINE (TRIPWIRE) UNTUK ZERO-DUPLICATE COUNTING
# =====================================================================
class CountingLine:
    def __init__(self, line_y_ratio=0.55, direction="both", buffer_px=15):
        """
        Garis Hitung Virtual (Tripwire) untuk traffic counting bebas duplikasi.

        Parameters:
        -----------
        line_y_ratio : float
            Posisi garis horizontal relatif terhadap tinggi frame (0.0 - 1.0). Default: 0.55.
        direction : str
            Arah yang dihitung: 'down' (masuk ke bawah), 'up' (ke atas), atau 'both'.
        buffer_px : int
            Toleransi zona garis (dalam pixel).
        """
        self.line_y_ratio = line_y_ratio
        self.direction = direction
        self.buffer_px = buffer_px

        self.total_in = 0
        self.total_out = 0
        self.counted_ids = set()
        self.counted_directions = {}  # id -> 'IN' / 'OUT'
        self.class_counts = {}

    def update(self, tracked_objects, frame_height):
        """
        Evaluasi apakah objek melintasi garis hitung pada frame ini.
        ID pelacakan tetap konstan sebelum dan sesudah melewati garis.
        """
        line_y = int(frame_height * self.line_y_ratio)

        for obj in tracked_objects:
            obj_id = obj['id']
            trajectory = obj['trajectory']
            cls_name = obj['class_name']

            if obj_id in self.counted_ids:
                continue

            if len(trajectory) >= 2:
                prev_y = trajectory[-2][1]
                curr_y = trajectory[-1][1]

                # Melintas ke bawah (Down / In)
                if prev_y < line_y <= curr_y:
                    if self.direction in ["down", "both"]:
                        self.total_in += 1
                        self.counted_ids.add(obj_id)
                        self.counted_directions[obj_id] = "IN"
                        cls_k = str(cls_name)
                        self.class_counts[cls_k] = self.class_counts.get(cls_k, 0) + 1

                # Melintas ke atas (Up / Out)
                elif prev_y > line_y >= curr_y:
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
            'line_y': line_y,
            'counted_ids': set(self.counted_ids),
            'counted_directions': dict(self.counted_directions)
        }

    def reset(self):
        self.total_in = 0
        self.total_out = 0
        self.counted_ids.clear()
        self.counted_directions.clear()
        self.class_counts.clear()


# Helper function untuk manipulasi list STrack
def joint_stracks(tlist_a, tlist_b):
    exists = {}
    res = []
    for t in tlist_a:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlist_b:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlist_a, tlist_b):
    stracks = {t.track_id: t for t in tlist_a}
    for t in tlist_b:
        stracks.pop(t.track_id, None)
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = compute_iou_matrix([t.tlbr for t in stracksa], [t.tlbr for t in stracksb])
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
