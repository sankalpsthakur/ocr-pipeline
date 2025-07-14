#!/usr/bin/env python3
"""Training Pipeline for JAX Denoising Models

Trains the JAX-based denoising models on synthetic degraded data for robust OCR.
Uses Noise2Noise approach with synthetic clean/noisy pairs.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from PIL import Image
import cv2
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pickle

# Import our components
from jax_denoising_adapter import LightweightUNet, QualityClassifier, DenoisingConfig
from synthetic_degradation import SyntheticDegradation


@dataclass
class TrainingConfig:
    """Configuration for JAX denoising training."""
    # Training parameters
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 8
    warmup_epochs: int = 10
    
    # Model parameters
    patch_size: int = 128
    overlap: int = 16
    
    # Data parameters
    num_training_pairs: int = 1000
    validation_split: float = 0.2
    
    # Augmentation parameters
    augment_probability: float = 0.8
    noise_levels: List[str] = None
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = "jax_checkpoints"
    
    # Device
    device: str = "cpu"
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = ['low', 'medium', 'high']


class JAXTrainingDataset:
    """Dataset for JAX denoising training."""
    
    def __init__(self, clean_images: List[np.ndarray], config: TrainingConfig):
        self.clean_images = clean_images
        self.config = config
        self.degrader = SyntheticDegradation()
        
        # Generate training pairs
        self.training_pairs = self._generate_training_pairs()
        
        # Split into train/validation
        split_idx = int(len(self.training_pairs) * (1 - config.validation_split))
        self.train_pairs = self.training_pairs[:split_idx]
        self.val_pairs = self.training_pairs[split_idx:]
        
        print(f"Generated {len(self.train_pairs)} training pairs")
        print(f"Generated {len(self.val_pairs)} validation pairs")
    
    def _generate_training_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate clean/degraded training pairs."""
        pairs = []
        
        pairs_per_image = self.config.num_training_pairs // len(self.clean_images)
        
        for clean_img in self.clean_images:
            for _ in range(pairs_per_image):
                # Random severity
                severity = np.random.choice(self.config.noise_levels)
                
                # Apply degradation
                degraded_img, _ = self.degrader.degrade_image(clean_img, severity=severity)
                
                pairs.append((clean_img.astype(np.float32) / 255.0, 
                             degraded_img.astype(np.float32) / 255.0))
        
        return pairs
    
    def get_batch(self, pairs: List[Tuple[np.ndarray, np.ndarray]], 
                  indices: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get a batch of training data."""
        batch_clean = []
        batch_noisy = []
        
        for idx in indices:
            clean, noisy = pairs[idx]
            
            # Extract random patches
            clean_patch, noisy_patch = self._extract_random_patch(clean, noisy)
            
            batch_clean.append(clean_patch)
            batch_noisy.append(noisy_patch)
        
        return jnp.array(batch_clean), jnp.array(batch_noisy)
    
    def _extract_random_patch(self, clean: np.ndarray, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random patch from image pair."""
        h, w = clean.shape[:2]
        patch_size = self.config.patch_size
        
        # Random top-left corner
        max_y = max(0, h - patch_size)
        max_x = max(0, w - patch_size)
        
        y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        
        # Extract patches
        clean_patch = clean[y:y+patch_size, x:x+patch_size]
        noisy_patch = noisy[y:y+patch_size, x:x+patch_size]
        
        # Pad if necessary
        if clean_patch.shape[0] < patch_size or clean_patch.shape[1] < patch_size:
            pad_h = max(0, patch_size - clean_patch.shape[0])
            pad_w = max(0, patch_size - clean_patch.shape[1])
            
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            noisy_patch = np.pad(noisy_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        return clean_patch, noisy_patch


class JAXDenoisingTrainer:
    """Trainer for JAX denoising models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.denoiser = LightweightUNet()
        self.quality_net = QualityClassifier()
        
        # Training states
        self.denoiser_state = None
        self.quality_state = None
        
        # Training history
        self.history = {
            'denoiser_loss': [],
            'quality_loss': [],
            'val_denoiser_loss': [],
            'val_quality_loss': []
        }
    
    def initialize_training(self, sample_batch: Tuple[jnp.ndarray, jnp.ndarray]):
        """Initialize training states."""
        clean_batch, noisy_batch = sample_batch
        
        # Initialize denoiser
        key = jax.random.PRNGKey(42)
        denoiser_params = self.denoiser.init(key, noisy_batch)
        
        # Create optimizer with warmup
        total_steps = self.config.num_epochs * (1000 // self.config.batch_size)  # Approximate
        warmup_steps = self.config.warmup_epochs * (1000 // self.config.batch_size)
        
        denoiser_tx = optax.chain(
            optax.linear_schedule(
                init_value=0.0,
                end_value=self.config.learning_rate,
                transition_steps=warmup_steps
            ),
            optax.adam(self.config.learning_rate)
        )
        
        self.denoiser_state = train_state.TrainState.create(
            apply_fn=self.denoiser.apply,
            params=denoiser_params,
            tx=denoiser_tx
        )
        
        # Initialize quality classifier
        # Create quality labels (dummy for now)
        dummy_quality = jnp.ones((clean_batch.shape[0], 3))  # 3 quality classes
        quality_params = self.quality_net.init(key, clean_batch)
        
        quality_tx = optax.adam(self.config.learning_rate)
        
        self.quality_state = train_state.TrainState.create(
            apply_fn=self.quality_net.apply,
            params=quality_params,
            tx=quality_tx
        )
        
        print("Training states initialized successfully")
    
    @jit
    def denoiser_loss_fn(self, params, noisy_batch, clean_batch):
        """Denoising loss function."""
        predicted = self.denoiser.apply(params, noisy_batch, training=True)
        
        # L1 + L2 loss combination
        l1_loss = jnp.mean(jnp.abs(predicted - clean_batch))
        l2_loss = jnp.mean((predicted - clean_batch) ** 2)
        
        # Perceptual loss (simplified - edge preservation)
        def edge_loss(pred, target):
            # Simple edge detection
            sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            # Apply to each channel
            pred_gray = jnp.mean(pred, axis=-1, keepdims=True)
            target_gray = jnp.mean(target, axis=-1, keepdims=True)
            
            return jnp.mean((pred_gray - target_gray) ** 2)
        
        edge_preservation = edge_loss(predicted, clean_batch)
        
        total_loss = l1_loss + 0.1 * l2_loss + 0.05 * edge_preservation
        
        return total_loss, {
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'edge_loss': edge_preservation,
            'total_loss': total_loss
        }
    
    @jit
    def quality_loss_fn(self, params, images, quality_labels):
        """Quality classification loss function."""
        predicted = self.quality_net.apply(params, images, training=True)
        
        # Cross-entropy loss
        log_probs = jnp.log(predicted + 1e-8)
        loss = -jnp.mean(jnp.sum(quality_labels * log_probs, axis=1))
        
        # Accuracy
        predicted_classes = jnp.argmax(predicted, axis=1)
        true_classes = jnp.argmax(quality_labels, axis=1)
        accuracy = jnp.mean(predicted_classes == true_classes)
        
        return loss, {'loss': loss, 'accuracy': accuracy}
    
    @jit
    def train_denoiser_step(self, state, noisy_batch, clean_batch):
        """Single training step for denoiser."""
        (loss, metrics), grads = value_and_grad(self.denoiser_loss_fn, has_aux=True)(
            state.params, noisy_batch, clean_batch
        )
        
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics
    
    @jit 
    def train_quality_step(self, state, images, quality_labels):
        """Single training step for quality classifier."""
        (loss, metrics), grads = value_and_grad(self.quality_loss_fn, has_aux=True)(
            state.params, images, quality_labels
        )
        
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics
    
    def train(self, dataset: JAXTrainingDataset):
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Training samples: {len(dataset.train_pairs)}")
        print(f"Validation samples: {len(dataset.val_pairs)}")
        
        # Initialize with sample batch
        sample_indices = np.arange(min(self.config.batch_size, len(dataset.train_pairs)))
        sample_batch = dataset.get_batch(dataset.train_pairs, sample_indices)
        self.initialize_training(sample_batch)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Training phase
            train_denoiser_losses = []
            train_quality_losses = []
            
            # Shuffle training data
            train_indices = np.random.permutation(len(dataset.train_pairs))
            
            for i in range(0, len(train_indices), self.config.batch_size):
                batch_indices = train_indices[i:i + self.config.batch_size]
                if len(batch_indices) < self.config.batch_size:
                    continue
                
                # Get batch
                clean_batch, noisy_batch = dataset.get_batch(dataset.train_pairs, batch_indices)
                
                # Train denoiser
                self.denoiser_state, denoiser_loss, denoiser_metrics = self.train_denoiser_step(
                    self.denoiser_state, noisy_batch, clean_batch
                )
                train_denoiser_losses.append(denoiser_loss)
                
                # Generate quality labels based on degradation level
                quality_labels = self._generate_quality_labels(clean_batch, noisy_batch)
                
                # Train quality classifier
                self.quality_state, quality_loss, quality_metrics = self.train_quality_step(
                    self.quality_state, noisy_batch, quality_labels
                )
                train_quality_losses.append(quality_loss)
            
            # Validation phase
            val_denoiser_losses = []
            val_quality_losses = []
            
            for i in range(0, len(dataset.val_pairs), self.config.batch_size):
                if i + self.config.batch_size > len(dataset.val_pairs):
                    break
                    
                batch_indices = np.arange(i, i + self.config.batch_size)
                clean_batch, noisy_batch = dataset.get_batch(dataset.val_pairs, batch_indices)
                
                # Validation denoiser loss
                val_denoiser_loss, _ = self.denoiser_loss_fn(
                    self.denoiser_state.params, noisy_batch, clean_batch
                )
                val_denoiser_losses.append(val_denoiser_loss)
                
                # Validation quality loss
                quality_labels = self._generate_quality_labels(clean_batch, noisy_batch)
                val_quality_loss, _ = self.quality_loss_fn(
                    self.quality_state.params, noisy_batch, quality_labels
                )
                val_quality_losses.append(val_quality_loss)
            
            # Calculate epoch metrics
            epoch_train_denoiser = np.mean(train_denoiser_losses)
            epoch_train_quality = np.mean(train_quality_losses)
            epoch_val_denoiser = np.mean(val_denoiser_losses) if val_denoiser_losses else 0
            epoch_val_quality = np.mean(val_quality_losses) if val_quality_losses else 0
            
            # Update history
            self.history['denoiser_loss'].append(epoch_train_denoiser)
            self.history['quality_loss'].append(epoch_train_quality)
            self.history['val_denoiser_loss'].append(epoch_val_denoiser)
            self.history['val_quality_loss'].append(epoch_val_quality)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Denoiser: {epoch_train_denoiser:.4f} (val: {epoch_val_denoiser:.4f}) | "
                  f"Quality: {epoch_train_quality:.4f} (val: {epoch_val_quality:.4f}) | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0 or epoch == self.config.num_epochs - 1:
                self.save_checkpoint(epoch + 1)
            
            # Save best model
            current_val_loss = epoch_val_denoiser + epoch_val_quality
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self.save_checkpoint(epoch + 1, is_best=True)
        
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return self.history
    
    def _generate_quality_labels(self, clean_batch: jnp.ndarray, noisy_batch: jnp.ndarray) -> jnp.ndarray:
        """Generate quality labels based on image degradation."""
        batch_size = clean_batch.shape[0]
        labels = []
        
        for i in range(batch_size):
            # Calculate PSNR as quality metric
            mse = jnp.mean((clean_batch[i] - noisy_batch[i]) ** 2)
            psnr = 20 * jnp.log10(1.0) - 10 * jnp.log10(mse + 1e-8)
            
            # Convert PSNR to quality class
            if psnr > 30:
                quality = [0, 0, 1]  # High quality
            elif psnr > 20:
                quality = [0, 1, 0]  # Medium quality  
            else:
                quality = [1, 0, 0]  # Low quality
            
            labels.append(quality)
        
        return jnp.array(labels)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_data = {
            'epoch': epoch,
            'denoiser_params': self.denoiser_state.params,
            'quality_params': self.quality_state.params,
            'denoiser_opt_state': self.denoiser_state.opt_state,
            'quality_opt_state': self.quality_state.opt_state,
            'history': self.history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pkl"
            with open(best_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"Saved best checkpoint at epoch {epoch}")
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Create training states with loaded parameters
        dummy_input = jnp.ones((1, 128, 128, 3))
        
        # Denoiser
        denoiser_tx = optax.adam(self.config.learning_rate)
        self.denoiser_state = train_state.TrainState.create(
            apply_fn=self.denoiser.apply,
            params=checkpoint_data['denoiser_params'],
            tx=denoiser_tx
        )
        self.denoiser_state = self.denoiser_state.replace(opt_state=checkpoint_data['denoiser_opt_state'])
        
        # Quality classifier
        quality_tx = optax.adam(self.config.learning_rate)
        self.quality_state = train_state.TrainState.create(
            apply_fn=self.quality_net.apply,
            params=checkpoint_data['quality_params'],
            tx=quality_tx
        )
        self.quality_state = self.quality_state.replace(opt_state=checkpoint_data['quality_opt_state'])
        
        self.history = checkpoint_data['history']
        
        print(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Denoiser loss
        epochs = range(1, len(self.history['denoiser_loss']) + 1)
        ax1.plot(epochs, self.history['denoiser_loss'], label='Train')
        ax1.plot(epochs, self.history['val_denoiser_loss'], label='Validation')
        ax1.set_title('Denoiser Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Quality classifier loss
        ax2.plot(epochs, self.history['quality_loss'], label='Train')
        ax2.plot(epochs, self.history['val_quality_loss'], label='Validation')
        ax2.set_title('Quality Classifier Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.checkpoint_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history saved to {plot_path}")


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
    print("JAX Denoising Model Training Pipeline")
    print("=" * 50)
    
    # Configuration
    config = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=50,  # Reduced for testing
        batch_size=4,   # Reduced for memory
        patch_size=128,
        num_training_pairs=200,  # Reduced for testing
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
    dataset = JAXTrainingDataset(clean_images, config)
    
    # Create trainer
    trainer = JAXDenoisingTrainer(config)
    
    # Train models
    print("\nStarting training...")
    history = trainer.train(dataset)
    
    # Plot results
    trainer.plot_training_history()
    
    # Save final results
    results = {
        'config': config.__dict__,
        'final_metrics': {
            'final_denoiser_loss': history['denoiser_loss'][-1],
            'final_quality_loss': history['quality_loss'][-1],
            'best_val_denoiser_loss': min(history['val_denoiser_loss']),
            'best_val_quality_loss': min(history['val_quality_loss'])
        },
        'history': history
    }
    
    results_path = trainer.checkpoint_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed successfully!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Best denoiser validation loss: {min(history['val_denoiser_loss']):.4f}")
    print(f"Best quality validation loss: {min(history['val_quality_loss']):.4f}")


if __name__ == "__main__":
    main()