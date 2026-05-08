"""
Model Training Pipeline

Handles training, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import json
from datetime import datetime


logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Training pipeline for Transformer model."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: str = 'models/checkpoints'):
        """
        Initialize training pipeline.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            device: Device to use ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_pir_loss': [],
            'train_risk_loss': [],
            'val_loss': [],
            'val_pir_loss': [],
            'val_risk_loss': [],
            'test_loss': None
        }
    
    def train(self,
              num_epochs: int = 50,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-5,
              pir_weight: float = 0.5,
              risk_weight: float = 0.5,
              patience: int = 10) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization weight
            pir_weight: Weight for PIR loss
            risk_weight: Weight for risk classification loss
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss functions
        pir_loss_fn = nn.MSELoss()
        risk_loss_fn = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_pir_loss, train_risk_loss = self._train_epoch(
                optimizer, pir_loss_fn, risk_loss_fn, pir_weight, risk_weight
            )
            
            # Validation phase
            val_loss, val_pir_loss, val_risk_loss = self._validate_epoch(
                pir_loss_fn, risk_loss_fn, pir_weight, risk_weight
            )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_pir_loss'].append(train_pir_loss)
            self.history['train_risk_loss'].append(train_risk_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_pir_loss'].append(val_pir_loss)
            self.history['val_risk_loss'].append(val_risk_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Logging
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Best Val Loss: {self.best_val_loss:.6f}"
                )
            
            # Early stopping and checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, optimizer, 'best')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        logger.info("Training completed")
        return self.history
    
    def _train_epoch(self, optimizer, pir_loss_fn, risk_loss_fn, 
                     pir_weight, risk_weight) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_pir_loss = 0.0
        total_risk_loss = 0.0
        
        for x, y_pir, y_risk in self.train_loader:
            x = x.to(self.device)
            y_pir = y_pir.to(self.device).unsqueeze(1)
            y_risk = y_risk.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            pir_pred, risk_logits = self.model(x)
            
            # Compute loss
            loss_pir = pir_loss_fn(pir_pred, y_pir)
            loss_risk = risk_loss_fn(risk_logits, y_risk)
            loss = pir_weight * loss_pir + risk_weight * loss_risk
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_pir_loss += loss_pir.item()
            total_risk_loss += loss_risk.item()
        
        n_batches = len(self.train_loader)
        return total_loss / n_batches, total_pir_loss / n_batches, total_risk_loss / n_batches
    
    def _validate_epoch(self, pir_loss_fn, risk_loss_fn,
                       pir_weight, risk_weight) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_pir_loss = 0.0
        total_risk_loss = 0.0
        
        with torch.no_grad():
            for x, y_pir, y_risk in self.val_loader:
                x = x.to(self.device)
                y_pir = y_pir.to(self.device).unsqueeze(1)
                y_risk = y_risk.to(self.device)
                
                # Forward pass
                pir_pred, risk_logits = self.model(x)
                
                # Compute loss
                loss_pir = pir_loss_fn(pir_pred, y_pir)
                loss_risk = risk_loss_fn(risk_logits, y_risk)
                loss = pir_weight * loss_pir + risk_weight * loss_risk
                
                # Accumulate losses
                total_loss += loss.item()
                total_pir_loss += loss_pir.item()
                total_risk_loss += loss_risk.item()
        
        n_batches = len(self.val_loader)
        return total_loss / n_batches, total_pir_loss / n_batches, total_risk_loss / n_batches
    
    def _save_checkpoint(self, epoch: int, optimizer, tag: str = 'best') -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{tag}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def evaluate(self) -> Dict:
        """Evaluate on test set."""
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return {}
        
        self.model.eval()
        
        pir_loss_fn = nn.MSELoss()
        risk_loss_fn = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_pir_loss = 0.0
        total_risk_loss = 0.0
        correct_risk = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y_pir, y_risk in self.test_loader:
                x = x.to(self.device)
                y_pir = y_pir.to(self.device).unsqueeze(1)
                y_risk = y_risk.to(self.device)
                
                # Forward pass
                pir_pred, risk_logits = self.model(x)
                
                # Compute metrics
                loss_pir = pir_loss_fn(pir_pred, y_pir)
                loss_risk = risk_loss_fn(risk_logits, y_risk)
                loss = 0.5 * loss_pir + 0.5 * loss_risk
                
                total_loss += loss.item()
                total_pir_loss += loss_pir.item()
                total_risk_loss += loss_risk.item()
                
                # Classification accuracy
                risk_pred = torch.argmax(risk_logits, dim=1)
                correct_risk += (risk_pred == y_risk).sum().item()
                total_samples += y_risk.size(0)
        
        n_batches = len(self.test_loader)
        
        results = {
            'test_loss': total_loss / n_batches,
            'test_pir_loss': total_pir_loss / n_batches,
            'test_risk_loss': total_risk_loss / n_batches,
            'test_risk_accuracy': correct_risk / total_samples
        }
        
        logger.info(f"Test Results: {results}")
        return results
    
    def save_history(self, save_path: str = 'training_history.json') -> None:
        """Save training history to JSON."""
        # Convert lists to ensure JSON serialization
        history_to_save = {
            key: value if value is None else list(value)
            for key, value in self.history.items()
        }
        
        with open(save_path, 'w') as f:
            json.dump(history_to_save, f, indent=2)
        logger.info(f"Training history saved: {save_path}")
