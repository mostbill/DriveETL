"""Anomaly detection module for AutoDataPipeline."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from config import ANOMALY_DETECTION_CONFIG
from .logging_config import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """AI-powered anomaly detection using multiple algorithms."""
    
    def __init__(self):
        """Initialize the anomaly detection module."""
        self.config = ANOMALY_DETECTION_CONFIG
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        logger.info("AnomalyDetector module initialized")
    
    def train_isolation_forest(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> None:
        """Train the Isolation Forest model.
        
        Args:
            df: Training DataFrame
            feature_columns: List of feature columns to use (auto-detect if None)
        """
        logger.info("Training Isolation Forest model")
        
        # Select feature columns
        if feature_columns is None:
            self.feature_columns = self._select_feature_columns(df)
        else:
            self.feature_columns = feature_columns
        
        logger.info(f"Using features: {self.feature_columns}")
        
        # Prepare training data
        X_train = df[self.feature_columns].copy()
        
        # Handle missing values
        X_train = X_train.fillna(X_train.mean())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train Isolation Forest
        iso_config = self.config["isolation_forest"]
        self.isolation_forest = IsolationForest(
            contamination=iso_config["contamination"],
            random_state=iso_config["random_state"],
            n_estimators=iso_config["n_estimators"]
        )
        
        self.isolation_forest.fit(X_train_scaled)
        self.is_trained = True
        
        logger.info(f"Isolation Forest trained on {len(X_train)} samples")
    
    def detect_anomalies(self, df: pd.DataFrame, method: str = 'combined') -> pd.DataFrame:
        """Detect anomalies in the dataset.
        
        Args:
            df: Input DataFrame
            method: Detection method ('isolation_forest', 'threshold', or 'combined')
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        logger.info(f"Detecting anomalies using {method} method")
        
        df_result = df.copy()
        
        if method in ['isolation_forest', 'combined']:
            if not self.is_trained:
                logger.warning("Isolation Forest not trained. Training on current data.")
                self.train_isolation_forest(df)
            
            df_result = self._detect_isolation_forest_anomalies(df_result)
        
        if method in ['threshold', 'combined']:
            df_result = self._detect_threshold_anomalies(df_result)
        
        if method == 'combined':
            df_result = self._combine_anomaly_predictions(df_result)
        
        # Calculate anomaly statistics
        self._log_anomaly_statistics(df_result)
        
        return df_result
    
    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Automatically select feature columns for anomaly detection."""
        # Exclude non-numeric and identifier columns
        exclude_columns = [
            'timestamp', 'vehicle_id', 'anomaly_isolation_forest', 
            'anomaly_threshold', 'anomaly_combined', 'anomaly_score'
        ]
        
        # Select numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Prioritize core sensor features
        core_features = ['speed', 'engine_temperature', 'fuel_level']
        available_core = [col for col in core_features if col in feature_columns]
        
        if available_core:
            # Use core features plus any rolling/derived features
            selected_features = available_core.copy()
            for col in feature_columns:
                if any(core in col for core in core_features) and col not in selected_features:
                    selected_features.append(col)
        else:
            selected_features = feature_columns
        
        return selected_features[:10]  # Limit to top 10 features
    
    def _detect_isolation_forest_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        # Prepare data
        X = df[self.feature_columns].copy()
        X = X.fillna(X.mean())
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies
        predictions = self.isolation_forest.predict(X_scaled)
        scores = self.isolation_forest.decision_function(X_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to boolean
        df['anomaly_isolation_forest'] = (predictions == -1)
        df['isolation_forest_score'] = scores
        
        return df
    
    def _detect_threshold_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using threshold-based rules."""
        threshold_config = self.config["threshold_detection"]
        
        # Initialize anomaly flags
        df['anomaly_threshold'] = False
        df['threshold_violations'] = ''
        
        violations = []
        
        # Speed threshold
        if 'speed' in df.columns:
            speed_anomalies = df['speed'] > threshold_config['speed_max']
            df.loc[speed_anomalies, 'anomaly_threshold'] = True
            df.loc[speed_anomalies, 'threshold_violations'] += 'speed_high;'
        
        # Engine temperature threshold
        if 'engine_temperature' in df.columns:
            temp_anomalies = df['engine_temperature'] > threshold_config['engine_temp_max']
            df.loc[temp_anomalies, 'anomaly_threshold'] = True
            df.loc[temp_anomalies, 'threshold_violations'] += 'temp_high;'
        
        # Fuel level threshold
        if 'fuel_level' in df.columns:
            fuel_anomalies = df['fuel_level'] < threshold_config['fuel_level_min']
            df.loc[fuel_anomalies, 'anomaly_threshold'] = True
            df.loc[fuel_anomalies, 'threshold_violations'] += 'fuel_low;'
        
        # Clean up violation strings
        df['threshold_violations'] = df['threshold_violations'].str.rstrip(';')
        
        return df
    
    def _combine_anomaly_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine multiple anomaly detection methods."""
        # Combine predictions (OR logic - anomaly if either method detects it)
        df['anomaly_combined'] = (
            df.get('anomaly_isolation_forest', False) | 
            df.get('anomaly_threshold', False)
        )
        
        # Create combined score
        iso_score = df.get('isolation_forest_score', 0)
        threshold_score = df['anomaly_threshold'].astype(int)
        
        # Normalize and combine scores
        df['anomaly_score'] = (iso_score * 0.7) + (threshold_score * 0.3)
        
        return df
    
    def _log_anomaly_statistics(self, df: pd.DataFrame) -> None:
        """Log statistics about detected anomalies."""
        total_records = len(df)
        
        if 'anomaly_isolation_forest' in df.columns:
            iso_anomalies = df['anomaly_isolation_forest'].sum()
            logger.info(f"Isolation Forest detected {iso_anomalies} anomalies ({iso_anomalies/total_records*100:.2f}%)")
        
        if 'anomaly_threshold' in df.columns:
            threshold_anomalies = df['anomaly_threshold'].sum()
            logger.info(f"Threshold method detected {threshold_anomalies} anomalies ({threshold_anomalies/total_records*100:.2f}%)")
        
        if 'anomaly_combined' in df.columns:
            combined_anomalies = df['anomaly_combined'].sum()
            logger.info(f"Combined method detected {combined_anomalies} anomalies ({combined_anomalies/total_records*100:.2f}%)")
    
    def evaluate_model(self, df: pd.DataFrame, true_anomaly_column: str) -> Dict[str, float]:
        """Evaluate anomaly detection performance.
        
        Args:
            df: DataFrame with predictions and true labels
            true_anomaly_column: Column name containing true anomaly labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if true_anomaly_column not in df.columns:
            logger.warning(f"True anomaly column '{true_anomaly_column}' not found")
            return {}
        
        y_true = df[true_anomaly_column]
        results = {}
        
        # Evaluate each method
        for method in ['anomaly_isolation_forest', 'anomaly_threshold', 'anomaly_combined']:
            if method in df.columns:
                y_pred = df[method]
                
                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                
                results[method] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'true_negatives': tn
                }
                
                logger.info(f"{method} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return results
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.config = model_data.get('config', self.config)
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores (if available).
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.feature_columns is None:
            logger.warning("Model not trained or feature columns not available")
            return None
        
        # Isolation Forest doesn't provide feature importance directly
        # We can use the decision function to estimate importance
        logger.info("Feature importance not directly available for Isolation Forest")
        return {col: 1.0 / len(self.feature_columns) for col in self.feature_columns}