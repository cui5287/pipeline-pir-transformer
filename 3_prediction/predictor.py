"""
Real-time PIR Predictor

Provides real-time prediction of PIR values based on input features.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class PIRPredictor:
    """Real-time PIR prediction."""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.model = None
        self.scaler_stats = {
            'x_mean': 0.5, 'x_std': 0.289,
            'p_mean': 5.0, 'p_std': 2.887,
            'd_mean': 650, 'd_std': 317.5,
            't_mean': 20, 't_std': 23.1,
            'pir_mean': 0.5, 'pir_std': 0.289
        }
        
        if Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning(f"Model path not found: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load model from checkpoint."""
        try:
            from .transformer_model import TransformerPIRModel
            
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = TransformerPIRModel().to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def normalize(self, x: np.ndarray, feature_name: str) -> np.ndarray:
        """Normalize input feature."""
        mean_key = f'{feature_name}_mean'
        std_key = f'{feature_name}_std'
        
        if mean_key in self.scaler_stats and std_key in self.scaler_stats:
            mean = self.scaler_stats[mean_key]
            std = self.scaler_stats[std_key]
            return (x - mean) / (std + 1e-8)
        
        return x
    
    def denormalize(self, x: np.ndarray, feature_name: str = 'pir') -> np.ndarray:
        """Denormalize output."""
        mean_key = f'{feature_name}_mean'
        std_key = f'{feature_name}_std'
        
        if mean_key in self.scaler_stats and std_key in self.scaler_stats:
            mean = self.scaler_stats[mean_key]
            std = self.scaler_stats[std_key]
            return x * std + mean
        
        return x
    
    def predict(self,
                x_sequence: np.ndarray,
                p_sequence: np.ndarray,
                d_sequence: np.ndarray,
                t_sequence: np.ndarray,
                q_sequence: np.ndarray = None) -> Dict:
        """
        Predict PIR value.
        
        Args:
            x_sequence: Hydrogen concentration sequence (seq_len,) or (seq_len,)
            p_sequence: Pressure sequence (seq_len,)
            d_sequence: Diameter sequence (seq_len,)
            t_sequence: Temperature sequence (seq_len,)
            q_sequence: Flow rate sequence (seq_len,) - optional
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {'error': 'Model not loaded'}
        
        try:
            # Normalize inputs
            x_norm = self.normalize(x_sequence, 'x')
            p_norm = self.normalize(p_sequence, 'p')
            d_norm = self.normalize(d_sequence, 'd')
            t_norm = self.normalize(t_sequence, 't')
            
            # Default flow rate if not provided
            if q_sequence is None:
                q_sequence = np.zeros_like(x_sequence)
            
            # Stack features: (seq_len, 5)
            x_input = np.stack([x_norm, p_norm, d_norm, t_norm, q_sequence], axis=1)
            
            # Convert to tensor and add batch dimension
            x_tensor = torch.from_numpy(x_input).float().unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                pir_pred, risk_logits = self.model(x_tensor)
            
            # Extract predictions
            pir_value = pir_pred.squeeze().cpu().numpy()
            risk_probs = torch.softmax(risk_logits, dim=1).squeeze().cpu().numpy()
            
            # Clip to valid range
            pir_value = np.clip(pir_value, 0.0, 1.0)
            
            return {
                'pir_value': float(pir_value),
                'pir_normalized': float(pir_value),
                'risk_probabilities': {
                    'low': float(risk_probs[0]),
                    'medium': float(risk_probs[1]),
                    'high': float(risk_probs[2])
                },
                'timestamp': None,
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def predict_future(self,
                      x_sequence: np.ndarray,
                      p_sequence: np.ndarray,
                      d_sequence: np.ndarray,
                      t_sequence: np.ndarray,
                      steps: int = 2) -> Dict:
        """
        Predict future PIR values (1-10 minutes ahead).
        
        Args:
            x_sequence: Historical hydrogen concentration
            p_sequence: Historical pressure
            d_sequence: Historical diameter
            t_sequence: Historical temperature
            steps: Number of steps to predict (1-2 for 5-10 minutes)
            
        Returns:
            Dictionary with future predictions
        """
        predictions = []
        current_x = x_sequence[-1]
        current_p = p_sequence[-1]
        current_d = d_sequence[-1]
        current_t = t_sequence[-1]
        
        for _ in range(steps):
            # Predict next step
            result = self.predict(
                x_sequence, p_sequence, d_sequence, t_sequence
            )
            
            if 'error' in result:
                break
            
            predictions.append(result['pir_value'])
            
            # Update sequences (simple approach: use last value)
            x_sequence = np.roll(x_sequence, -1)
            x_sequence[-1] = current_x
            
            p_sequence = np.roll(p_sequence, -1)
            p_sequence[-1] = current_p
            
            d_sequence = np.roll(d_sequence, -1)
            d_sequence[-1] = current_d
            
            t_sequence = np.roll(t_sequence, -1)
            t_sequence[-1] = current_t
        
        return {
            'future_predictions': predictions,
            'forecast_minutes': [5 * (i + 1) for i in range(len(predictions))],
            'status': 'success' if predictions else 'failed'
        }
    
    def batch_predict(self, batch_inputs: List[Dict]) -> List[Dict]:
        """
        Batch prediction for multiple samples.
        
        Args:
            batch_inputs: List of input dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for inputs in batch_inputs:
            result = self.predict(
                inputs['x_sequence'],
                inputs['p_sequence'],
                inputs['d_sequence'],
                inputs['t_sequence'],
                inputs.get('q_sequence', None)
            )
            results.append(result)
        
        return results
