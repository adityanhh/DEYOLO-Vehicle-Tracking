"""
Deep Appearance Feature Extractor (Re-ID) untuk DeepSORT
==========================================================
Mengekstrak visual appearance descriptor 128-dimensi yang dinormalisasi L2 (||f||=1)
dari potongan gambar (cropped bounding box) kendaraan.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDNet(nn.Module):
    """
    Arsitektur CNN Ringan untuk Ekstraksi Ciri Penampilan Objek (Re-ID Appearance Descriptor).
    Output: Vektor Fitur 128-Dimensi.
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        features = self.fc(x)
        # Normalisasi L2: ||features||_2 = 1 (Penting untuk Cosine Distance)
        return F.normalize(features, p=2, dim=1)


class FeatureExtractor:
    def __init__(self, feature_dim=128, input_size=(128, 64), device=None):
        """
        Parameters:
        -----------
        feature_dim : int
            Dimensi embedding vektor penampilan (default: 128).
        input_size : tuple (height, width)
            Ukuran resize crop kendaraan (default: 128x64).
        device : str or torch.device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.input_size = input_size
        self.net = ReIDNet(feature_dim=feature_dim).to(self.device)
        self.net.eval()

        # Nilai normalisasi ImageNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def extract(self, frame, bboxes):
        """
        Ekstraksi feature embeddings untuk semua bounding box pada frame.

        Parameters:
        -----------
        frame : np.ndarray
            Citra frame BGR dari OpenCV (H, W, 3).
        bboxes : list of [x1, y1, x2, y2]
            Koordinat bounding box objek.

        Returns:
        --------
        np.ndarray:
            Matriks fitur shape (N, feature_dim).
        """
        if len(bboxes) == 0:
            return np.empty((0, 128), dtype=np.float32)

        h_img, w_img = frame.shape[:2]
        patches = []

        for bbox in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Clamp koordinat ke batas dimensi gambar
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(x1 + 1, min(x2, w_img))
            y2 = max(y1 + 1, min(y2, h_img))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                crop = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))

            # BGR ke RGB dan normalisasi
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop_norm = (crop_rgb - self.mean) / self.std
            # Transpose (H, W, C) -> (C, H, W)
            patches.append(np.transpose(crop_norm, (2, 0, 1)))

        batch_tensor = torch.tensor(np.array(patches), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            features = self.net(batch_tensor).cpu().numpy()

        return features
