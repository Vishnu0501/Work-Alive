"""
Machine Learning Model Module

Implements ML models for activity classification and provides
training and prediction capabilities.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import logging

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from .logging_setup import get_logger
from .activity_monitor import ActivityData


class ActivityClassifier:
    """ML model for classifying user activity as working or idle."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the activity classifier.
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.logger = get_logger(__name__)
        self.model_path = Path(model_path) if model_path else None
        
        # Models
        self.classifier: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Feature names for interpretability
        self.feature_names = [
            'keyboard_events',
            'mouse_events', 
            'mouse_distance',
            'screen_change_score',
            'active_window_changes',
            'cpu_usage',
            'memory_usage'
        ]
        
        # Model metadata
        self.model_info = {
            'trained_at': None,
            'training_samples': 0,
            'accuracy': 0.0,
            'version': '1.0'
        }
        
        # Load existing model if available
        if self.model_path and self.model_path.exists():
            self.load_model()
            
    def create_synthetic_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic training data for initial model training.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features, labels)
        """
        self.logger.info(f"Generating {n_samples} synthetic training samples")
        
        np.random.seed(42)  # For reproducibility
        
        # Generate working activity patterns
        working_samples = n_samples // 2
        
        # Working patterns: high keyboard/mouse activity, screen changes
        working_keyboard = np.random.normal(25, 10, working_samples)  # 15-35 events
        working_mouse = np.random.normal(40, 15, working_samples)     # 25-55 events  
        working_mouse_dist = np.random.normal(800, 300, working_samples)  # 500-1100 pixels
        working_screen = np.random.beta(2, 3, working_samples) * 0.8  # 0.1-0.8 change score
        working_windows = np.random.poisson(3, working_samples)       # 0-6 window changes
        working_cpu = np.random.normal(35, 15, working_samples)       # 20-50% CPU
        working_memory = np.random.normal(60, 20, working_samples)    # 40-80% memory
        
        # Idle patterns: low activity across all metrics
        idle_samples = n_samples - working_samples
        
        idle_keyboard = np.random.exponential(2, idle_samples)        # 0-10 events
        idle_mouse = np.random.exponential(5, idle_samples)           # 0-20 events
        idle_mouse_dist = np.random.exponential(100, idle_samples)    # 0-300 pixels
        idle_screen = np.random.beta(5, 20, idle_samples) * 0.3       # 0-0.1 change score
        idle_windows = np.random.poisson(0.5, idle_samples)           # 0-2 window changes
        idle_cpu = np.random.normal(15, 8, idle_samples)              # 5-25% CPU
        idle_memory = np.random.normal(45, 15, idle_samples)          # 30-60% memory
        
        # Combine features
        features = np.vstack([
            np.column_stack([
                working_keyboard, working_mouse, working_mouse_dist,
                working_screen, working_windows, working_cpu, working_memory
            ]),
            np.column_stack([
                idle_keyboard, idle_mouse, idle_mouse_dist,
                idle_screen, idle_windows, idle_cpu, idle_memory
            ])
        ])
        
        # Create labels (1 = working, 0 = idle)
        labels = np.hstack([
            np.ones(working_samples),
            np.zeros(idle_samples)
        ])
        
        # Add some noise and ensure non-negative values
        features = np.maximum(features + np.random.normal(0, 0.1, features.shape), 0)
        
        # Shuffle the data
        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]
        
        return features, labels
        
    def train_model(self, 
                   features: Optional[np.ndarray] = None, 
                   labels: Optional[np.ndarray] = None,
                   use_synthetic: bool = True) -> Dict[str, float]:
        """
        Train the activity classification model.
        
        Args:
            features: Training features (optional)
            labels: Training labels (optional)
            use_synthetic: Whether to use synthetic data if no real data provided
            
        Returns:
            Training metrics dictionary
        """
        self.logger.info("Training activity classification model")
        
        # Use synthetic data if no real data provided
        if features is None or labels is None:
            if use_synthetic:
                features, labels = self.create_synthetic_training_data()
            else:
                raise ValueError("No training data provided and synthetic data disabled")
                
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train main classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.classifier.fit(X_train_scaled, y_train)
        
        # Train anomaly detector for outlier detection
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        # Train only on working samples for anomaly detection
        working_samples = X_train_scaled[y_train == 1]
        if len(working_samples) > 10:
            self.anomaly_detector.fit(working_samples)
        
        # Evaluate model
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train_scaled, y_train, cv=5)
        
        # Predictions for detailed metrics
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Update model info
        self.model_info.update({
            'trained_at': datetime.now(),
            'training_samples': len(features),
            'accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
        
        # Log results
        self.logger.info(f"Model training completed:")
        self.logger.info(f"  Training accuracy: {train_score:.3f}")
        self.logger.info(f"  Test accuracy: {test_score:.3f}")
        self.logger.info(f"  CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.classifier.feature_importances_))
        self.logger.info("Feature importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {feature}: {importance:.3f}")
            
        # Save model
        if self.model_path:
            self.save_model()
            
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
        
    def predict(self, features: np.ndarray, return_confidence: bool = True) -> Tuple[int, float]:
        """
        Predict activity class for given features.
        
        Args:
            features: Feature vector
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Tuple of (prediction, confidence) where prediction is 1 for working, 0 for idle
        """
        if self.classifier is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        prediction = self.classifier.predict(features_scaled)[0]
        
        # Get confidence (probability)
        if return_confidence:
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
        else:
            confidence = 0.0
            
        return int(prediction), confidence
        
    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict activity classes for batch of features.
        
        Args:
            features: Feature matrix (n_samples x n_features)
            
        Returns:
            Tuple of (predictions, confidences)
        """
        if self.classifier is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
        
    def detect_anomaly(self, features: np.ndarray) -> bool:
        """
        Detect if the activity pattern is anomalous.
        
        Args:
            features: Feature vector
            
        Returns:
            True if anomalous, False if normal
        """
        if self.anomaly_detector is None:
            return False
            
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict anomaly (-1 for anomaly, 1 for normal)
        anomaly_score = self.anomaly_detector.predict(features_scaled)[0]
        
        return anomaly_score == -1
        
    def save_model(self) -> None:
        """Save the trained model to disk."""
        if self.model_path is None:
            raise ValueError("No model path specified")
            
        if self.classifier is None:
            raise ValueError("No trained model to save")
            
        # Ensure directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'classifier': self.classifier,
            'anomaly_detector': self.anomaly_detector,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_info': self.model_info
        }
        
        joblib.dump(model_data, self.model_path)
        self.logger.info(f"Model saved to {self.model_path}")
        
    def load_model(self) -> None:
        """Load a trained model from disk."""
        if self.model_path is None or not self.model_path.exists():
            raise ValueError(f"Model file not found: {self.model_path}")
            
        try:
            model_data = joblib.load(self.model_path)
            
            self.classifier = model_data['classifier']
            self.anomaly_detector = model_data.get('anomaly_detector')
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', self.feature_names)
            self.model_info = model_data.get('model_info', self.model_info)
            
            self.logger.info(f"Model loaded from {self.model_path}")
            self.logger.info(f"Model trained at: {self.model_info.get('trained_at')}")
            self.logger.info(f"Model accuracy: {self.model_info.get('accuracy', 'Unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def is_model_trained(self) -> bool:
        """Check if the model is trained and ready for predictions."""
        return self.classifier is not None and self.scaler is not None
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        return self.model_info.copy()
        
    def retrain_if_needed(self, max_age_days: int = 7) -> bool:
        """
        Check if model needs retraining based on age.
        
        Args:
            max_age_days: Maximum age of model in days before retraining
            
        Returns:
            True if model was retrained, False otherwise
        """
        if not self.is_model_trained():
            self.logger.info("No trained model found, training new model")
            self.train_model()
            return True
            
        trained_at = self.model_info.get('trained_at')
        if trained_at is None:
            self.logger.info("Model training date unknown, retraining")
            self.train_model()
            return True
            
        age = datetime.now() - trained_at
        if age.days >= max_age_days:
            self.logger.info(f"Model is {age.days} days old, retraining")
            self.train_model()
            return True
            
        return False
        
    def explain_prediction(self, features: np.ndarray) -> Dict[str, float]:
        """
        Explain a prediction by showing feature contributions.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary mapping feature names to their importance scores
        """
        if not self.is_model_trained():
            raise ValueError("Model not trained")
            
        # Get feature importances from the trained model
        importances = self.classifier.feature_importances_
        
        # Normalize features to show relative contributions
        features_normalized = features / (np.abs(features) + 1e-8)
        
        # Calculate feature contributions
        contributions = {}
        for i, (feature_name, importance) in enumerate(zip(self.feature_names, importances)):
            contribution = importance * abs(features_normalized[i])
            contributions[feature_name] = contribution
            
        return contributions
        
    def generate_training_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive training report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        if not self.is_model_trained():
            return "No trained model available for reporting."
            
        report = []
        report.append("Activity Classification Model Report")
        report.append("=" * 40)
        report.append(f"Model Version: {self.model_info.get('version', 'Unknown')}")
        report.append(f"Trained At: {self.model_info.get('trained_at', 'Unknown')}")
        report.append(f"Training Samples: {self.model_info.get('training_samples', 'Unknown')}")
        report.append(f"Test Accuracy: {self.model_info.get('accuracy', 'Unknown'):.3f}")
        report.append(f"CV Accuracy: {self.model_info.get('cv_mean', 'Unknown'):.3f} "
                     f"(±{self.model_info.get('cv_std', 0):.3f})")
        report.append("")
        
        # Feature importance
        report.append("Feature Importance:")
        report.append("-" * 20)
        importances = dict(zip(self.feature_names, self.classifier.feature_importances_))
        for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            report.append(f"{feature:20}: {importance:.3f}")
            
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text
