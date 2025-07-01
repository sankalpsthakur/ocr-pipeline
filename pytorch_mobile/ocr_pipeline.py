"""PyTorch Mobile OCR Pipeline - All-in-one implementation.

This file contains the complete PyTorch OCR pipeline optimized for mobile deployment.
It includes models, preprocessing, postprocessing, and the main pipeline logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
import numpy as np
from PIL import Image
import cv2
import re
import time
import logging
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Model Architectures
# ============================================================================

class MobileNetV3Block(nn.Module):
    """MobileNetV3 building block with squeeze-and-excitation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, expand_ratio: int = 1, se_ratio: float = 0.25,
                 activation: str = 'hswish'):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expand_ratio)
        self.use_expansion = expand_ratio != 1
        
        layers = []
        if self.use_expansion:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                self._get_activation(activation)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            self._get_activation(activation)
        ])
        
        if se_ratio > 0:
            se_channels = int(in_channels * se_ratio)
            layers.append(SqueezeExcitation(hidden_dim, se_channels))
        
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def _get_activation(self, name: str):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'hswish':
            return nn.Hardswish(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class SqueezeExcitation(nn.Module):
    """Squeeze-and-excitation module for channel attention."""
    
    def __init__(self, in_channels: int, se_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, se_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(se_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DBHead(nn.Module):
    """Differentiable Binarization head for text detection."""
    
    def __init__(self, in_channels: int, k: int = 50):
        super().__init__()
        self.k = k
        
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )
        
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        prob_map = self.binarize(x)
        
        if self.training:
            thresh_map = self.thresh(x)
            binary_map = torch.sigmoid(self.k * (prob_map - thresh_map))
            return {'prob_map': prob_map, 'thresh_map': thresh_map, 'binary_map': binary_map}
        else:
            return prob_map


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""
    
    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) 
            for _ in in_channels
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        fpn_features = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        
        target_size = fpn_features[0].shape[2:]
        merged = fpn_features[0]
        for feat in fpn_features[1:]:
            merged += F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
        
        return merged


class TextDetector(nn.Module):
    """Mobile-optimized text detection model based on DBNet architecture."""
    
    def __init__(self):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            
            MobileNetV3Block(16, 16, 3, 1, 1, se_ratio=0.25),
            MobileNetV3Block(16, 24, 3, 2, 4, se_ratio=0),
            MobileNetV3Block(24, 24, 3, 1, 3, se_ratio=0),
            
            MobileNetV3Block(24, 40, 5, 2, 3, se_ratio=0.25),
            MobileNetV3Block(40, 40, 5, 1, 3, se_ratio=0.25),
            MobileNetV3Block(40, 40, 5, 1, 3, se_ratio=0.25),
            
            MobileNetV3Block(40, 80, 3, 2, 6, se_ratio=0),
            MobileNetV3Block(80, 80, 3, 1, 2.5, se_ratio=0),
            MobileNetV3Block(80, 80, 3, 1, 2.3, se_ratio=0),
            MobileNetV3Block(80, 80, 3, 1, 2.3, se_ratio=0),
            
            MobileNetV3Block(80, 112, 3, 1, 6, se_ratio=0.25),
            MobileNetV3Block(112, 112, 3, 1, 6, se_ratio=0.25),
            
            MobileNetV3Block(112, 160, 5, 2, 6, se_ratio=0.25),
            MobileNetV3Block(160, 160, 5, 1, 6, se_ratio=0.25),
            MobileNetV3Block(160, 160, 5, 1, 6, se_ratio=0.25),
        )
        
        self.fpn = FPN(in_channels=[24, 40, 80, 160], out_channels=256)
        self.det_head = DBHead(256)
    
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [3, 6, 10, 16]:
                features.append(x)
        
        fused_features = self.fpn(features)
        det_results = self.det_head(fused_features)
        
        return det_results


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for sequence modeling."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        recurrent, _ = self.lstm(x)
        output = self.linear(recurrent)
        return output


class TextRecognizer(nn.Module):
    """Mobile-optimized text recognition model based on CRNN architecture."""
    
    def __init__(self, num_classes: int = 96):
        super().__init__()
        self.num_classes = num_classes
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(512, 512, 2, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_classes)
        )
    
    def forward(self, x):
        conv_features = self.backbone(x)
        b, c, h, w = conv_features.size()
        assert h == 1, f"Height should be 1, got {h}"
        
        conv_features = conv_features.squeeze(2)
        conv_features = conv_features.permute(0, 2, 1)
        
        output = self.rnn(conv_features)
        return output


class AngleClassifier(nn.Module):
    """Lightweight model for text angle classification."""
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            MobileNetV3Block(32, 32, 3, 1, 1, se_ratio=0),
            MobileNetV3Block(32, 64, 3, 2, 4, se_ratio=0),
            MobileNetV3Block(64, 64, 3, 1, 3, se_ratio=0),
            MobileNetV3Block(64, 128, 5, 2, 3, se_ratio=0.25),
            MobileNetV3Block(128, 128, 5, 1, 3, se_ratio=0.25),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


# ============================================================================
# Preprocessing and Postprocessing
# ============================================================================

class ImagePreprocessor:
    """Handles image preprocessing for OCR models."""
    
    def __init__(self, target_height: int = 32, min_width: int = 32, 
                 max_width: int = 640, mean: List[float] = None, 
                 std: List[float] = None):
        self.target_height = target_height
        self.min_width = min_width
        self.max_width = max_width
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
    
    def preprocess_for_detection(self, image: Union[Image.Image, np.ndarray], 
                               target_size: int = 640) -> Tuple[torch.Tensor, float]:
        """Preprocess image for text detection model."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        w, h = image.size
        scale = target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32
        
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        img_tensor = torch.from_numpy(np.array(image)).float()
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(2).repeat(1, 1, 3)
        
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        
        for i in range(3):
            img_tensor[i] = (img_tensor[i] - self.mean[i]) / self.std[i]
        
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), value=0)
        
        return img_tensor.unsqueeze(0), scale
    
    def preprocess_for_recognition(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for text recognition model."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        w, h = image.size
        aspect_ratio = w / h
        new_h = self.target_height
        new_w = int(aspect_ratio * new_h)
        
        new_w = max(self.min_width, min(new_w, self.max_width))
        
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        img_tensor = torch.from_numpy(np.array(image)).float()
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(2).repeat(1, 1, 3)
        
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        
        for i in range(3):
            img_tensor[i] = (img_tensor[i] - self.mean[i]) / self.std[i]
        
        if new_w < self.max_width:
            pad_width = self.max_width - new_w
            img_tensor = F.pad(img_tensor, (0, pad_width, 0, 0), value=0)
        
        return img_tensor.unsqueeze(0)
    
    def preprocess_for_angle(self, image: Union[Image.Image, np.ndarray], 
                           target_size: Tuple[int, int] = (192, 48)) -> torch.Tensor:
        """Preprocess image for angle classification model."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        img_tensor = torch.from_numpy(np.array(image)).float()
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(2).repeat(1, 1, 3)
        
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        
        for i in range(3):
            img_tensor[i] = (img_tensor[i] - self.mean[i]) / self.std[i]
        
        return img_tensor.unsqueeze(0)


class DetectionPostProcessor:
    """Handles postprocessing for text detection results."""
    
    def __init__(self, thresh: float = 0.3, box_thresh: float = 0.5, 
                 max_candidates: int = 1000, unclip_ratio: float = 1.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
    
    def __call__(self, pred_map: torch.Tensor, scale: float) -> List[List[Tuple[int, int]]]:
        """Convert probability map to text boxes."""
        segmentation = self._binarize(pred_map)
        boxes = self._boxes_from_bitmap(segmentation)
        boxes = self._filter_boxes(boxes, scale)
        return boxes
    
    def _binarize(self, pred: torch.Tensor) -> np.ndarray:
        """Binarize prediction map."""
        pred_np = pred.squeeze().cpu().numpy()
        return (pred_np > self.thresh).astype(np.uint8)
    
    def _boxes_from_bitmap(self, bitmap: np.ndarray) -> List[np.ndarray]:
        """Extract boxes from binary map."""
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours[:self.max_candidates]:
            if len(contour) < 4:
                continue
            
            points = contour.reshape(-1, 2)
            score = self._box_score_fast(bitmap, points)
            
            if score < self.box_thresh:
                continue
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            box = self._unclip(box)
            boxes.append(box)
        
        return boxes
    
    def _box_score_fast(self, bitmap: np.ndarray, box: np.ndarray) -> float:
        """Calculate box score from bitmap."""
        h, w = bitmap.shape[:2]
        box = box.copy()
        
        xmin = np.clip(np.min(box[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(box[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(box[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(box[:, 1]), 0, h - 1)
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, [box.reshape(-1, 1, 2)], 1)
        
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def _unclip(self, box: np.ndarray) -> np.ndarray:
        """Expand box by unclip ratio."""
        poly = box.reshape(-1, 2)
        distance = cv2.contourArea(poly) * self.unclip_ratio / cv2.arcLength(poly, True)
        offset = cv2.dilate(
            poly.reshape(1, -1, 2).astype(np.float32),
            np.ones((3, 3), np.uint8),
            iterations=int(distance)
        )
        return offset.reshape(-1, 2).astype(np.int32)
    
    def _filter_boxes(self, boxes: List[np.ndarray], scale: float) -> List[List[Tuple[int, int]]]:
        """Filter and scale boxes."""
        filtered_boxes = []
        
        for box in boxes:
            box = box / scale
            box_points = [(int(x), int(y)) for x, y in box]
            
            area = cv2.contourArea(np.array(box_points))
            if area > 10:
                filtered_boxes.append(box_points)
        
        return filtered_boxes


class RecognitionPostProcessor:
    """Handles postprocessing for text recognition results."""
    
    def __init__(self, use_ctc_decode: bool = True):
        self.use_ctc_decode = use_ctc_decode
        self.character_dict = self._load_character_dict()
        self.blank_idx = len(self.character_dict)
    
    def _load_character_dict(self) -> Dict[int, str]:
        """Load character dictionary."""
        chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")
        return {i: char for i, char in enumerate(chars)}
    
    def __call__(self, preds: torch.Tensor) -> List[Tuple[str, float]]:
        """Decode recognition predictions."""
        if self.use_ctc_decode:
            return self._ctc_decode(preds)
        else:
            return self._greedy_decode(preds)
    
    def _ctc_decode(self, preds: torch.Tensor) -> List[Tuple[str, float]]:
        """CTC decoding for sequence predictions."""
        preds_np = preds.cpu().numpy()
        texts = []
        
        for pred in preds_np:
            pred_idx = np.argmax(pred, axis=1)
            
            char_list = []
            confidence_list = []
            prev_idx = self.blank_idx
            
            for i, idx in enumerate(pred_idx):
                if idx != self.blank_idx and idx != prev_idx:
                    char_list.append(self.character_dict.get(idx, ''))
                    confidence_list.append(pred[i, idx])
                prev_idx = idx
            
            text = ''.join(char_list)
            confidence = np.mean(confidence_list) if confidence_list else 0.0
            texts.append((text, float(confidence)))
        
        return texts
    
    def _greedy_decode(self, preds: torch.Tensor) -> List[Tuple[str, float]]:
        """Simple greedy decoding."""
        preds_np = preds.cpu().numpy()
        texts = []
        
        for pred in preds_np:
            pred_idx = np.argmax(pred, axis=1)
            char_list = [self.character_dict.get(idx, '') for idx in pred_idx]
            confidence_list = [pred[i, idx] for i, idx in enumerate(pred_idx)]
            
            text = ''.join(char_list).strip()
            confidence = np.mean(confidence_list) if confidence_list else 0.0
            texts.append((text, float(confidence)))
        
        return texts


class CharacterCorrector:
    """Handles character-level error correction for OCR results."""
    
    def __init__(self):
        self.numeric_corrections = {
            'l': '1', 'I': '1', '|': '1',
            'O': '0', 'o': '0',
            'Z': '2', 'z': '2',
            'S': '5', 's': '5',
            'G': '6', 'g': '9',
            'B': '8', 'b': '8'
        }
    
    def correct_text(self, text: str, is_numeric_context: bool = False) -> str:
        """Apply character corrections based on context."""
        if not is_numeric_context:
            return text
        
        corrected = []
        for char in text:
            if char in self.numeric_corrections:
                corrected.append(self.numeric_corrections[char])
            else:
                corrected.append(char)
        
        return ''.join(corrected)


# ============================================================================
# Helper Functions
# ============================================================================

def crop_text_region(image: np.ndarray, box: List[Tuple[int, int]]) -> np.ndarray:
    """Crop text region from image using perspective transform."""
    points = np.array(box, dtype=np.float32)
    
    width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3])
    ))
    height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])
    ))
    
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(points, dst_points)
    cropped = cv2.warpPerspective(image, M, (width, height))
    
    return cropped


def adjust_box_order(box: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Ensure box points are in clockwise order starting from top-left."""
    points = np.array(box)
    start_idx = np.argmin(points[:, 0] + points[:, 1])
    
    ordered_points = np.zeros_like(points)
    for i in range(4):
        ordered_points[i] = points[(start_idx + i) % 4]
    
    return [(int(x), int(y)) for x, y in ordered_points]


# ============================================================================
# Main Pipeline
# ============================================================================

@dataclass
class TorchOCRResult:
    """Result container for PyTorch OCR pipeline."""
    text: str
    boxes: List[List[Tuple[int, int]]]
    texts: List[str]
    confidences: List[float]
    processing_time: float
    
    @property
    def field_confidence(self) -> float:
        """Overall confidence score."""
        if not self.confidences:
            return 0.0
        valid_confs = [max(c, 1e-3) for c in self.confidences]
        product = np.prod(valid_confs)
        return float(product ** (1 / len(valid_confs)))


class TorchOCR:
    """PyTorch-based OCR engine for mobile deployment."""
    
    def __init__(self, 
                 det_model_path: Optional[str] = None,
                 rec_model_path: Optional[str] = None,
                 cls_model_path: Optional[str] = None,
                 use_angle_cls: bool = True,
                 device: str = 'cpu',
                 use_gpu: bool = False):
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else device)
        self.use_angle_cls = use_angle_cls
        
        # Initialize models
        self.det_model = self._load_detection_model(det_model_path)
        self.rec_model = self._load_recognition_model(rec_model_path)
        self.cls_model = self._load_angle_classifier(cls_model_path) if use_angle_cls else None
        
        # Initialize processors
        self.preprocessor = ImagePreprocessor()
        self.det_postprocessor = DetectionPostProcessor()
        self.rec_postprocessor = RecognitionPostProcessor()
        self.char_corrector = CharacterCorrector()
        
        LOGGER.info(f"TorchOCR initialized on {self.device}")
    
    def _load_detection_model(self, model_path: Optional[str]) -> TextDetector:
        """Load text detection model."""
        if model_path and Path(model_path).exists():
            model = torch.jit.load(model_path, map_location=self.device)
            LOGGER.info(f"Loaded detection model from {model_path}")
        else:
            model = TextDetector().to(self.device)
            model.eval()
            LOGGER.info("Using default detection model")
        return model
    
    def _load_recognition_model(self, model_path: Optional[str]) -> TextRecognizer:
        """Load text recognition model."""
        if model_path and Path(model_path).exists():
            model = torch.jit.load(model_path, map_location=self.device)
            LOGGER.info(f"Loaded recognition model from {model_path}")
        else:
            model = TextRecognizer().to(self.device)
            model.eval()
            LOGGER.info("Using default recognition model")
        return model
    
    def _load_angle_classifier(self, model_path: Optional[str]) -> AngleClassifier:
        """Load angle classification model."""
        if model_path and Path(model_path).exists():
            model = torch.jit.load(model_path, map_location=self.device)
            LOGGER.info(f"Loaded angle classifier from {model_path}")
        else:
            model = AngleClassifier().to(self.device)
            model.eval()
            LOGGER.info("Using default angle classifier")
        return model
    
    def ocr(self, image: Union[str, Path, Image.Image, np.ndarray]) -> TorchOCRResult:
        """Perform OCR on an image."""
        start_time = time.time()
        
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to numpy for processing
        img_np = np.array(image)
        
        # Detect text regions
        boxes = self._detect_text(image)
        
        if not boxes:
            LOGGER.warning("No text detected in image")
            return TorchOCRResult(
                text="",
                boxes=[],
                texts=[],
                confidences=[],
                processing_time=time.time() - start_time
            )
        
        # Recognize text in each box
        texts = []
        confidences = []
        
        for box in boxes:
            # Crop text region
            box = adjust_box_order(box)
            cropped = crop_text_region(img_np, box)
            
            # Angle classification if enabled
            if self.use_angle_cls:
                angle = self._classify_angle(cropped)
                if angle != 0:
                    cropped = self._rotate_image(cropped, angle)
            
            # Recognize text
            text, confidence = self._recognize_text(cropped)
            
            # Apply character correction
            is_numeric = self._is_numeric_context(text)
            if is_numeric:
                text = self.char_corrector.correct_text(text, is_numeric_context=True)
            
            texts.append(text)
            confidences.append(confidence)
        
        # Combine all texts
        full_text = ' '.join(texts)
        
        processing_time = time.time() - start_time
        LOGGER.info(f"OCR completed in {processing_time:.3f}s")
        
        return TorchOCRResult(
            text=full_text,
            boxes=boxes,
            texts=texts,
            confidences=confidences,
            processing_time=processing_time
        )
    
    def _detect_text(self, image: Image.Image) -> List[List[Tuple[int, int]]]:
        """Detect text regions in image."""
        img_tensor, scale = self.preprocessor.preprocess_for_detection(image)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.det_model(img_tensor)
        
        if isinstance(output, dict):
            pred_map = output.get('prob_map', output.get('binary_map', output))
        else:
            pred_map = output
        
        boxes = self.det_postprocessor(pred_map, scale)
        return boxes
    
    def _recognize_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Recognize text in a cropped region."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_tensor = self.preprocessor.preprocess_for_recognition(image)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.rec_model(img_tensor)
        
        results = self.rec_postprocessor(output)
        
        if results:
            return results[0]
        else:
            return "", 0.0
    
    def _classify_angle(self, image: np.ndarray) -> int:
        """Classify text angle (0, 90, 180, 270)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_tensor = self.preprocessor.preprocess_for_angle(image)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.cls_model(img_tensor)
        
        angle_idx = torch.argmax(output, dim=1).item()
        angles = [0, 90, 180, 270]
        
        return angles[angle_idx]
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle."""
        if angle == 90:
            return np.rot90(image, k=-1)
        elif angle == 180:
            return np.rot90(image, k=2)
        elif angle == 270:
            return np.rot90(image, k=1)
        else:
            return image
    
    def _is_numeric_context(self, text: str) -> bool:
        """Check if text is in numeric context."""
        digits = sum(c.isdigit() for c in text)
        return digits > len(text) * 0.5


def extract_fields(text: str) -> Dict[str, str]:
    """Extract key fields from OCR text."""
    fields = {}
    
    # Electricity consumption patterns
    electricity_patterns = [
        r"(?:Electricity|Kilowatt\s*Hours?)[\s:]*(\d{1,4})\s*(?:kWh)?",
        r"Total\s*Consumption[\s:]*(\d{1,4})\s*kWh",
        r"(\d{1,4})\s*kWh"
    ]
    
    for pattern in electricity_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if 50 <= int(value) <= 9999:
                fields['electricity_kwh'] = value
                break
    
    # Carbon footprint patterns
    carbon_patterns = [
        r"Carbon\s*Footprint[:\s]*(\d{1,4})\s*(?:kg\s*CO2e?)?",
        r"(\d{1,4})\s*[Kk][Gg]\s*CO2e?",
        r"CO2e?\s*[:=]\s*(\d{1,4})"
    ]
    
    for pattern in carbon_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if 10 <= int(value) <= 9999:
                fields['carbon_kgco2e'] = value
                break
    
    return fields


def run_ocr(image_path: Union[str, Path]) -> TorchOCRResult:
    """Main entry point for OCR processing."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Initialize OCR engine
    ocr_engine = TorchOCR(use_gpu=torch.cuda.is_available())
    
    # Process image
    result = ocr_engine.ocr(image_path)
    
    return result


def run_ocr_with_fields(image_path: Union[str, Path]) -> Dict[str, any]:
    """Run OCR and extract fields."""
    # Run OCR
    ocr_result = run_ocr(image_path)
    
    # Extract fields
    fields = extract_fields(ocr_result.text)
    
    # Add metadata
    fields['_ocr_confidence'] = ocr_result.field_confidence
    fields['_processing_time'] = ocr_result.processing_time
    fields['_full_text'] = ocr_result.text
    
    return fields


# ============================================================================
# Mobile Export Functions
# ============================================================================

def quantize_model(model: nn.Module, model_type: str = 'detection') -> nn.Module:
    """Quantize a model for mobile deployment."""
    model.eval()
    
    # Prepare quantization config
    model.qconfig = quantization.get_default_qconfig('qnnpack')
    
    # Create quantizable version
    if model_type == 'detection':
        quant_model = QuantizedTextDetector(model)
    elif model_type == 'recognition':
        quant_model = QuantizedTextRecognizer(model)
    elif model_type == 'angle':
        quant_model = QuantizedAngleClassifier(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Prepare and convert
    quantization.prepare(quant_model, inplace=True)
    quantization.convert(quant_model, inplace=True)
    
    return quant_model


class QuantizedTextDetector(nn.Module):
    """Quantized version of TextDetector."""
    
    def __init__(self, base_model: TextDetector):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.backbone = base_model.backbone
        self.fpn = base_model.fpn
        self.det_head = base_model.det_head
    
    def forward(self, x):
        x = self.quant(x)
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [3, 6, 10, 16]:
                features.append(x)
        
        fused_features = self.fpn(features)
        det_results = self.det_head(fused_features)
        
        if isinstance(det_results, dict):
            for k, v in det_results.items():
                det_results[k] = self.dequant(v)
        else:
            det_results = self.dequant(det_results)
        
        return det_results


class QuantizedTextRecognizer(nn.Module):
    """Quantized version of TextRecognizer."""
    
    def __init__(self, base_model: TextRecognizer):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.backbone = base_model.backbone
        self.rnn = base_model.rnn
    
    def forward(self, x):
        x = self.quant(x)
        conv_features = self.backbone(x)
        b, c, h, w = conv_features.size()
        conv_features = conv_features.squeeze(2)
        conv_features = conv_features.permute(0, 2, 1)
        output = self.rnn(conv_features)
        output = self.dequant(output)
        return output


class QuantizedAngleClassifier(nn.Module):
    """Quantized version of AngleClassifier."""
    
    def __init__(self, base_model: AngleClassifier):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.backbone = base_model.backbone
        self.classifier = base_model.classifier
    
    def forward(self, x):
        x = self.quant(x)
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        output = self.dequant(output)
        return output


def export_models_for_mobile(output_dir: str = "mobile_models"):
    """Export models for mobile deployment."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create models
    det_model = TextDetector()
    rec_model = TextRecognizer()
    cls_model = AngleClassifier()
    
    # Set to eval mode
    det_model.eval()
    rec_model.eval()
    cls_model.eval()
    
    # Example inputs for tracing
    det_input = torch.randn(1, 3, 640, 640)
    rec_input = torch.randn(1, 3, 32, 320)
    cls_input = torch.randn(1, 3, 48, 192)
    
    # Trace and save models
    traced_det = torch.jit.trace(det_model, det_input)
    traced_rec = torch.jit.trace(rec_model, rec_input)
    traced_cls = torch.jit.trace(cls_model, cls_input)
    
    traced_det.save(os.path.join(output_dir, "text_detector.pt"))
    traced_rec.save(os.path.join(output_dir, "text_recognizer.pt"))
    traced_cls.save(os.path.join(output_dir, "angle_classifier.pt"))
    
    # Export to ONNX
    torch.onnx.export(det_model, det_input, 
                      os.path.join(output_dir, "text_detector.onnx"),
                      export_params=True, opset_version=11,
                      input_names=['input'], output_names=['output'])
    
    torch.onnx.export(rec_model, rec_input,
                      os.path.join(output_dir, "text_recognizer.onnx"),
                      export_params=True, opset_version=11,
                      input_names=['input'], output_names=['output'])
    
    torch.onnx.export(cls_model, cls_input,
                      os.path.join(output_dir, "angle_classifier.onnx"),
                      export_params=True, opset_version=11,
                      input_names=['input'], output_names=['output'])
    
    print(f"Models exported to {output_dir}/")
    
    # Print sizes
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"{filename}: {size_mb:.2f} MB")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        
        # Run OCR
        result = run_ocr_with_fields(image_path)
        
        # Print results
        print(f"\nElectricity: {result.get('electricity_kwh', 'Not found')} kWh")
        print(f"Carbon: {result.get('carbon_kgco2e', 'Not found')} kg CO2e")
        print(f"Confidence: {result.get('_ocr_confidence', 0):.3f}")
        print(f"Processing time: {result.get('_processing_time', 0):.3f}s")
    else:
        print("Usage: python ocr_pipeline.py <image_path>")