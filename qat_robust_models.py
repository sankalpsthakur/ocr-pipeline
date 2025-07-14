#!/usr/bin/env python3
"""Quantization Aware Training (QAT) for Robust OCR Models

Implements QAT-aware versions of OCR models with built-in noise robustness.
Uses PyTorch's fake quantization to simulate mobile deployment while training
for maximum accuracy on degraded images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
import torch.quantization as quantization
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import time


class NoiseAugmentationLayer(nn.Module):
    """Layer that adds noise during training for robustness."""
    
    def __init__(self, noise_type: str = 'gaussian', noise_prob: float = 0.5):
        super().__init__()
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        
    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.noise_prob:
            return x
        
        if self.noise_type == 'gaussian':
            # Add Gaussian noise
            noise_std = torch.rand(1).item() * 0.1  # Random std up to 0.1
            noise = torch.randn_like(x) * noise_std
            return torch.clamp(x + noise, 0, 1)
        
        elif self.noise_type == 'dropout':
            # Random dropout of pixels
            dropout_prob = torch.rand(1).item() * 0.2  # Up to 20% dropout
            mask = torch.rand_like(x) > dropout_prob
            return x * mask.float()
        
        elif self.noise_type == 'salt_pepper':
            # Salt and pepper noise
            noise_intensity = torch.rand(1).item() * 0.1
            salt_mask = torch.rand_like(x) < noise_intensity / 2
            pepper_mask = torch.rand_like(x) < noise_intensity / 2
            
            result = x.clone()
            result[salt_mask] = 1.0  # Salt (white)
            result[pepper_mask] = 0.0  # Pepper (black)
            return result
        
        return x


class QATConvBlock(nn.Module):
    """QAT-aware convolution block with noise robustness."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 use_noise: bool = True):
        super().__init__()
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Noise augmentation
        if use_noise:
            self.noise_layer = NoiseAugmentationLayer()
        else:
            self.noise_layer = nn.Identity()
        
        # Convolution with proper padding
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size, stride, 
            padding=kernel_size//2,
            bias=False
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Activation (ReLU for quantization compatibility)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.quant(x)
        x = self.noise_layer(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dequant(x)
        return x


class QATMobileNetV3Block(nn.Module):
    """QAT-aware MobileNetV3 block with noise robustness."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 expand_ratio: int = 1, se_ratio: float = 0.0):
        super().__init__()
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expand_ratio)
        self.use_expansion = expand_ratio != 1
        
        layers = []
        
        # Noise augmentation at input
        layers.append(NoiseAugmentationLayer(noise_prob=0.3))
        
        # Expansion
        if self.use_expansion:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.extend([
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_channels, hidden_dim, 1),
                nn.Sigmoid()
            ])
        
        # Pointwise
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.quant(x)
        out = self.block(x)
        
        if self.use_residual:
            out = out + x
        
        out = self.dequant(out)
        return out


class QATRobustTextDetector(nn.Module):
    """QAT-aware text detector with noise robustness."""
    
    def __init__(self, num_classes: int = 1):
        super().__init__()
        
        # Input quantization
        self.quant_input = QuantStub()
        
        # Robust backbone with noise augmentation
        self.backbone = nn.Sequential(
            # Initial conv with noise robustness
            QATConvBlock(3, 16, 3, 2, use_noise=True),
            QATConvBlock(16, 16, 3, 1, use_noise=True),
            
            # MobileNetV3 blocks with built-in noise
            QATMobileNetV3Block(16, 24, 3, 2, 4, 0.25),
            QATMobileNetV3Block(24, 24, 3, 1, 3, 0.25),
            QATMobileNetV3Block(24, 40, 5, 2, 3, 0.25),
            QATMobileNetV3Block(40, 40, 5, 1, 3, 0.25),
            QATMobileNetV3Block(40, 40, 5, 1, 3, 0.25),
            QATMobileNetV3Block(40, 80, 3, 2, 6, 0.25),
            QATMobileNetV3Block(80, 80, 3, 1, 2.5, 0.25),
            QATMobileNetV3Block(80, 80, 3, 1, 2.3, 0.25),
            QATMobileNetV3Block(80, 80, 3, 1, 2.3, 0.25),
            QATMobileNetV3Block(80, 112, 3, 1, 6, 0.25),
            QATMobileNetV3Block(112, 112, 3, 1, 6, 0.25),
            QATMobileNetV3Block(112, 160, 5, 2, 6, 0.25),
            QATMobileNetV3Block(160, 160, 5, 1, 6, 0.25),
            QATMobileNetV3Block(160, 160, 5, 1, 6, 0.25)
        )
        
        # Feature Pyramid Network (simplified)
        self.fpn = nn.ModuleList([
            QATConvBlock(24, 256),   # P2
            QATConvBlock(40, 256),   # P3
            QATConvBlock(80, 256),   # P4
            QATConvBlock(160, 256),  # P5
        ])
        
        # Detection head
        self.det_head = nn.Sequential(
            QATConvBlock(256, 64, use_noise=False),
            QATConvBlock(64, 64, use_noise=False),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Output dequantization
        self.dequant_output = DeQuantStub()
        
    def forward(self, x):
        x = self.quant_input(x)
        
        # Extract features at different scales
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # Extract features after specific layers
            if i in [2, 4, 7, 14]:  # After specific MobileNet blocks
                features.append(x)
        
        # Apply FPN
        fpn_features = []
        for feat, fpn_layer in zip(features, self.fpn):
            fpn_features.append(fpn_layer(feat))
        
        # Upsample and combine features
        target_size = fpn_features[0].shape[2:]
        combined = fpn_features[0]
        
        for feat in fpn_features[1:]:
            upsampled = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            combined = combined + upsampled
        
        # Detection head
        output = self.det_head(combined)
        output = self.dequant_output(output)
        
        return {'prob_map': torch.sigmoid(output)}


class QATRobustTextRecognizer(nn.Module):
    """QAT-aware text recognizer with noise robustness."""
    
    def __init__(self, vocab_size: int = 95, hidden_size: int = 256):
        super().__init__()
        
        self.quant_input = QuantStub()
        
        # Robust CNN backbone
        self.backbone = nn.Sequential(
            QATConvBlock(3, 32, use_noise=True),
            nn.MaxPool2d(2, 2),
            QATConvBlock(32, 64, use_noise=True),
            nn.MaxPool2d(2, 2),
            QATConvBlock(64, 128, use_noise=True),
            nn.MaxPool2d(2, 2),
            QATConvBlock(128, 256, use_noise=True),
            nn.MaxPool2d(2, 2),
            QATConvBlock(256, 512, use_noise=False)  # No noise in final layers
        )
        
        # LSTM for sequence modeling (quantization-friendly)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism (simplified for quantization)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output projection
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        
        self.dequant_output = DeQuantStub()
        
    def forward(self, x):
        x = self.quant_input(x)
        
        # CNN features
        conv_features = self.backbone(x)
        b, c, h, w = conv_features.size()
        
        # Reshape for RNN
        conv_features = conv_features.view(b, c, h * w)
        conv_features = conv_features.permute(0, 2, 1)
        
        # RNN processing
        rnn_output, _ = self.rnn(conv_features)
        
        # Simple attention (mean pooling for quantization compatibility)
        attended = torch.mean(rnn_output, dim=1, keepdim=True)
        
        # Classification
        output = self.classifier(attended)
        output = self.dequant_output(output)
        
        return output


class QATRobustOCRPipeline:
    """Complete QAT-aware OCR pipeline with noise robustness."""
    
    def __init__(self, detector_path: Optional[str] = None, 
                 recognizer_path: Optional[str] = None):
        self.detector = QATRobustTextDetector()
        self.recognizer = QATRobustTextRecognizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move to device
        self.detector.to(self.device)
        self.recognizer.to(self.device)
        
        # Load weights if provided
        if detector_path and Path(detector_path).exists():
            self.detector.load_state_dict(torch.load(detector_path, map_location=self.device))
        
        if recognizer_path and Path(recognizer_path).exists():
            self.recognizer.load_state_dict(torch.load(recognizer_path, map_location=self.device))
    
    def _remove_qat_modules(self):
        """Remove QAT modules if quantization fails."""
        def remove_qat_recursive(module):
            for name, child in module.named_children():
                if hasattr(child, 'quant') or hasattr(child, 'dequant'):
                    # Replace with regular modules
                    if hasattr(child, 'conv'):
                        setattr(module, name, child.conv)
                    elif hasattr(child, 'linear'):
                        setattr(module, name, child.linear)
                else:
                    remove_qat_recursive(child)
        
        remove_qat_recursive(self.detector)
        remove_qat_recursive(self.recognizer)
        print("QAT modules removed, using FP32 models")
    
    def prepare_for_qat_training(self):
        """Prepare models for QAT training."""
        # Set QAT config
        qat_config = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare models
        self.detector.qconfig = qat_config
        self.recognizer.qconfig = qat_config
        
        # Convert to QAT mode
        self.detector = prepare_qat(self.detector)
        self.recognizer = prepare_qat(self.recognizer)
        
        print("Models prepared for QAT training")
    
    def convert_to_quantized(self):
        """Convert trained QAT models to quantized inference."""
        self.detector.eval()
        self.recognizer.eval()
        
        try:
            # Convert to quantized models
            self.detector = convert(self.detector)
            self.recognizer = convert(self.recognizer)
            print("Models converted to quantized inference")
        except Exception as e:
            print(f"Quantization conversion failed: {e}")
            print("Using FP32 models instead")
            # Remove QAT modules if conversion fails
            self._remove_qat_modules()
    
    def train_detector_step(self, images: torch.Tensor, targets: torch.Tensor, 
                          optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Single training step for detector."""
        self.detector.train()
        
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.detector(images)
        loss = criterion(outputs['prob_map'], targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train_recognizer_step(self, images: torch.Tensor, texts: torch.Tensor,
                            optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Single training step for recognizer."""
        self.recognizer.train()
        
        images = images.to(self.device)
        texts = texts.to(self.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.recognizer(images)
        loss = criterion(outputs, texts)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def inference(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run inference on image."""
        self.detector.eval()
        self.recognizer.eval()
        
        with torch.no_grad():
            image = image.to(self.device)
            
            # Text detection
            det_output = self.detector(image)
            
            # For now, return detection output
            # Full pipeline would include text region extraction and recognition
            return det_output
    
    def benchmark_quantization(self, test_images: torch.Tensor, num_runs: int = 10):
        """Benchmark quantized vs full precision models."""
        print("Quantization Benchmark")
        print("=" * 40)
        
        # Test full precision
        self.detector.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.detector(test_images.to(self.device))
        
        fp32_time = (time.time() - start_time) / num_runs
        
        # Convert to quantized
        self.convert_to_quantized()
        
        # Test quantized
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.detector(test_images.to(self.device))
        
        int8_time = (time.time() - start_time) / num_runs
        
        # Calculate model sizes
        def get_model_size(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            return param_size / (1024 * 1024)  # MB
        
        # Create fresh models for size comparison
        fp32_model = QATRobustTextDetector()
        self.convert_to_quantized()  # Current model is quantized
        
        fp32_size = get_model_size(fp32_model)
        int8_size = get_model_size(self.detector)
        
        print(f"FP32 inference time: {fp32_time*1000:.2f}ms")
        print(f"INT8 inference time: {int8_time*1000:.2f}ms")
        print(f"Speedup: {fp32_time/int8_time:.2f}x")
        print(f"FP32 model size: {fp32_size:.2f}MB")
        print(f"INT8 model size: {int8_size:.2f}MB")
        print(f"Size reduction: {fp32_size/int8_size:.2f}x")
    
    def export_for_mobile(self, output_dir: str):
        """Export quantized models for mobile deployment."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure models are quantized
        self.convert_to_quantized()
        
        # Export with TorchScript
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        
        # Trace detector
        traced_detector = torch.jit.trace(self.detector, dummy_input)
        traced_detector.save(output_path / "qat_detector_quantized.pt")
        
        # Trace recognizer
        dummy_rec_input = torch.randn(1, 3, 32, 320).to(self.device)
        traced_recognizer = torch.jit.trace(self.recognizer, dummy_rec_input)
        traced_recognizer.save(output_path / "qat_recognizer_quantized.pt")
        
        print(f"Quantized models exported to {output_dir}")
        
        # Print file sizes
        for file in output_path.glob("*.pt"):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"{file.name}: {size_mb:.2f}MB")


def create_noise_robust_training_data(clean_images: List[torch.Tensor], 
                                     augmentation_factor: int = 5) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create training pairs with synthetic noise for robustness."""
    from synthetic_degradation import SyntheticDegradation
    
    degrader = SyntheticDegradation()
    training_pairs = []
    
    for clean_img in clean_images:
        # Convert to numpy for degradation
        clean_np = (clean_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        for _ in range(augmentation_factor):
            # Apply random degradations
            degraded_np, _ = degrader.degrade_image(clean_np)
            
            # Convert back to tensor
            degraded_tensor = torch.from_numpy(degraded_np).permute(2, 0, 1).float() / 255.0
            
            training_pairs.append((clean_img, degraded_tensor))
    
    return training_pairs


# Example usage
if __name__ == "__main__":
    # Initialize QAT pipeline
    pipeline = QATRobustOCRPipeline()
    
    # Prepare for QAT training
    pipeline.prepare_for_qat_training()
    
    print("QAT-aware OCR pipeline initialized")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Test inference
    output = pipeline.inference(dummy_input)
    print(f"Detector output shape: {output['prob_map'].shape}")
    
    # Benchmark quantization
    test_batch = torch.randn(4, 3, 640, 640)
    pipeline.benchmark_quantization(test_batch)
    
    # Export for mobile
    pipeline.export_for_mobile("qat_mobile_models")