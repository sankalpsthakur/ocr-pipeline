#!/usr/bin/env python3
"""QAT Training Pipeline for Robust OCR Models

Trains QAT-aware OCR models with synthetic degraded data for noise robustness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import our components
from qat_robust_models import QATRobustOCRPipeline, create_noise_robust_training_data
from synthetic_degradation import SyntheticDegradation


@dataclass
class QATTrainingConfig:
    """Configuration for QAT training."""
    # Training parameters
    learning_rate: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 4
    warmup_epochs: int = 5
    
    # Model parameters
    image_size: Tuple[int, int] = (640, 640)
    
    # Data parameters
    num_training_pairs: int = 200
    validation_split: float = 0.2
    
    # Augmentation parameters
    augmentation_factor: int = 5
    noise_levels: List[str] = None
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = "qat_checkpoints"
    
    # Device
    device: str = "auto"
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = ['low', 'medium', 'high']
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class QATTrainingDataset(Dataset):
    """Dataset for QAT training with synthetic degradations."""
    
    def __init__(self, clean_images: List[np.ndarray], config: QATTrainingConfig):
        self.clean_images = clean_images
        self.config = config
        self.degrader = SyntheticDegradation()
        
        # Generate training pairs
        self.training_pairs = self._generate_training_pairs()
        
        print(f"Generated {len(self.training_pairs)} training pairs")
    
    def _generate_training_pairs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate clean/degraded training pairs."""
        pairs = []
        
        pairs_per_image = self.config.num_training_pairs // len(self.clean_images)
        
        for clean_img in self.clean_images:
            for _ in range(pairs_per_image):
                # Random severity
                severity = np.random.choice(self.config.noise_levels)
                
                # Apply degradation
                degraded_img, _ = self.degrader.degrade_image(clean_img, severity=severity)
                
                # Convert to tensors
                clean_tensor = self._image_to_tensor(clean_img)
                degraded_tensor = self._image_to_tensor(degraded_img)
                
                pairs.append((clean_tensor, degraded_tensor))
        
        return pairs
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor and resize."""
        # Resize to target size
        h, w = self.config.image_size
        resized = cv2.resize(image, (w, h))
        
        # Convert to tensor [C, H, W] and normalize
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        
        return tensor
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        clean, degraded = self.training_pairs[idx]
        return {
            'clean': clean,
            'degraded': degraded,
            'target': self._create_dummy_target()  # Placeholder for detection target
        }
    
    def _create_dummy_target(self) -> torch.Tensor:
        """Create dummy detection target for training."""
        # For now, create a simple binary mask (1 where text should be)
        # In production, would use real ground truth annotations
        h, w = self.config.image_size
        target = torch.zeros(1, h // 8, w // 8)  # Downsampled target
        
        # Add some random "text regions" for training
        center_h, center_w = h // 16, w // 16
        target[:, center_h-5:center_h+5, center_w-10:center_w+10] = 1.0
        
        return target


class QATTrainer:
    """Trainer for QAT-aware OCR models."""
    
    def __init__(self, config: QATTrainingConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline
        self.pipeline = QATRobustOCRPipeline()
        self.pipeline.prepare_for_qat_training()
        
        # Initialize optimizers
        self.detector_optimizer = optim.Adam(
            self.pipeline.detector.parameters(), 
            lr=config.learning_rate
        )
        
        # Loss functions
        self.detector_criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.history = {
            'detector_loss': [],
            'val_detector_loss': []
        }
    
    def train(self, dataset: QATTrainingDataset):
        """Main training loop."""
        print(f"Starting QAT training for {self.config.num_epochs} epochs")
        print(f"Device: {self.config.device}")
        print(f"Batch size: {self.config.batch_size}")
        
        # Split dataset
        split_idx = int(len(dataset) * (1 - self.config.validation_split))
        train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
        val_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            
            # Update history
            self.history['detector_loss'].append(train_loss)
            self.history['val_detector_loss'].append(val_loss)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0 or epoch == self.config.num_epochs - 1:
                self.save_checkpoint(epoch + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, is_best=True)
        
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.pipeline.detector.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Use degraded images as input
            images = batch['degraded'].to(self.config.device)
            targets = batch['target'].to(self.config.device)
            
            # Train detector
            loss = self.pipeline.train_detector_step(
                images, targets, self.detector_optimizer, self.detector_criterion
            )
            
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.pipeline.detector.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['degraded'].to(self.config.device)
                targets = batch['target'].to(self.config.device)
                
                # Forward pass
                outputs = self.pipeline.detector(images)
                loss = self.detector_criterion(outputs['prob_map'], targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_data = {
            'epoch': epoch,
            'detector_state_dict': self.pipeline.detector.state_dict(),
            'detector_optimizer_state_dict': self.detector_optimizer.state_dict(),
            'history': self.history,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"qat_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "qat_best_checkpoint.pt"
            torch.save(checkpoint_data, best_path)
            print(f"Saved best checkpoint at epoch {epoch}")
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.pipeline.detector.load_state_dict(checkpoint['detector_state_dict'])
        self.detector_optimizer.load_state_dict(checkpoint['detector_optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def convert_to_quantized_and_export(self):
        """Convert to quantized models and export."""
        print("Converting to quantized models...")
        self.pipeline.convert_to_quantized()
        
        # Export for mobile
        self.pipeline.export_for_mobile(str(self.checkpoint_dir / "mobile_models"))
        
        # Benchmark quantization
        dummy_batch = torch.randn(2, 3, *self.config.image_size)
        self.pipeline.benchmark_quantization(dummy_batch)


def load_training_images(image_paths: List[str]) -> List[np.ndarray]:
    """Load images for training."""
    images = []
    
    for img_path in image_paths:
        if Path(img_path).exists():
            image = Image.open(img_path)
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            images.append(img_array)
            print(f"Loaded {img_path}: {img_array.shape}")
    
    return images


def main():
    """Main training function."""
    print("QAT Robust OCR Training Pipeline")
    print("=" * 50)
    
    # Configuration
    config = QATTrainingConfig(
        learning_rate=1e-3,
        num_epochs=30,  # Reduced for testing
        batch_size=2,   # Reduced for memory
        image_size=(320, 320),  # Smaller for faster training
        num_training_pairs=50,  # Reduced for testing
        save_every=10
    )
    
    # Load training images
    image_paths = []
    for img_name in ['DEWA.png', 'SEWA.png']:
        if Path(img_name).exists():
            image_paths.append(img_name)
    
    if not image_paths:
        print("No training images found! Please ensure DEWA.png and SEWA.png are available.")
        return
    
    clean_images = load_training_images(image_paths)
    
    if len(clean_images) == 0:
        print("No valid images loaded!")
        return
    
    # Create dataset
    print(f"\nCreating training dataset with {config.num_training_pairs} pairs...")
    dataset = QATTrainingDataset(clean_images, config)
    
    # Create trainer
    trainer = QATTrainer(config)
    
    # Train models
    print("\nStarting QAT training...")
    history = trainer.train(dataset)
    
    # Convert to quantized and export
    print("\nConverting to quantized models...")
    trainer.convert_to_quantized_and_export()
    
    # Save final results
    results = {
        'config': config.__dict__,
        'final_metrics': {
            'final_train_loss': history['detector_loss'][-1],
            'final_val_loss': history['val_detector_loss'][-1],
            'best_val_loss': min(history['val_detector_loss'])
        },
        'history': history
    }
    
    results_path = trainer.checkpoint_dir / "qat_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nQAT training completed successfully!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Best validation loss: {min(history['val_detector_loss']):.4f}")


if __name__ == "__main__":
    main()