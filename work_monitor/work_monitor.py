"""
Main Work Monitor Module

Orchestrates all components to provide comprehensive activity monitoring.
"""

import time
import threading
from datetime import datetime, time as dt_time, timedelta
from typing import Optional, Dict, Any
import signal
import sys

from .config import ConfigManager
from .logging_setup import get_logger
from .activity_monitor import ActivityMonitor
from .ml_model import ActivityClassifier
from .email_service import EmailService


class WorkMonitor:
    """Main work monitoring system that coordinates all components."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the work monitor.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Component initialization
        self.activity_monitor: Optional[ActivityMonitor] = None
        self.ml_classifier: Optional[ActivityClassifier] = None
        self.email_service: Optional[EmailService] = None
        
        # State management
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Activity tracking
        self.last_activity_check = datetime.now()
        self.consecutive_idle_periods = 0
        self.daily_stats = {
            'work_minutes': 0,
            'idle_minutes': 0,
            'total_keyboard_events': 0,
            'total_mouse_events': 0,
            'alerts_sent': 0,
            'start_time': datetime.now()
        }
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize all monitoring components."""
        try:
            # Initialize activity monitor
            self.activity_monitor = ActivityMonitor(
                check_interval=self.config.get('monitoring.check_interval_seconds', 30),
                screenshot_interval=self.config.get('monitoring.screenshot_interval_seconds', 60)
            )
            
            # Initialize ML classifier
            model_path = self.config.get_model_path()
            self.ml_classifier = ActivityClassifier(str(model_path))
            
            # Train model if needed
            if not self.ml_classifier.is_model_trained():
                self.logger.info("No trained model found, training new model")
                self.ml_classifier.train_model()
            else:
                # Check if retraining is needed
                retrain_interval = self.config.get('model.retrain_interval_days', 7)
                self.ml_classifier.retrain_if_needed(retrain_interval)
            
            # Initialize email service if configured
            if self.config.validate_email_config():
                self.email_service = EmailService(
                    smtp_server=self.config.get('email.smtp_server'),
                    smtp_port=self.config.get('email.smtp_port'),
                    sender_email=self.config.get('email.sender_email'),
                    sender_password=self.config.get('email.sender_password'),
                    recipient_email=self.config.get('email.recipient_email')
                )
                self.logger.info("Email notifications enabled")
            else:
                self.logger.warning("Email configuration invalid - notifications disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
            
    def start(self) -> None:
        """Start the work monitoring system."""
        if self.is_running:
            self.logger.warning("Work monitor is already running")
            return
            
        self.logger.info("Starting work monitor")
        
        try:
            # Start activity monitoring
            self.activity_monitor.start_monitoring()
            
            # Start main monitoring loop
            self.is_running = True
            self.stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Work monitor started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start work monitor: {e}")
            self.stop()
            raise
            
    def stop(self) -> None:
        """Stop the work monitoring system."""
        if not self.is_running:
            return
            
        self.logger.info("Stopping work monitor")
        
        # Signal stop
        self.is_running = False
        self.stop_event.set()
        
        # Stop activity monitoring
        if self.activity_monitor:
            self.activity_monitor.stop_monitoring()
            
        # Wait for monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
            
        # Send daily summary if email is configured
        if self.email_service:
            self._send_daily_summary()
            
        self.logger.info("Work monitor stopped")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs continuously."""
        self.logger.info("Monitoring loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Check if we're in working hours
                if not self._is_working_hours():
                    self.logger.debug("Outside working hours, skipping monitoring")
                    self.stop_event.wait(60)  # Check every minute
                    continue
                    
                # Perform activity analysis
                self._analyze_activity()
                
                # Update daily statistics
                self._update_daily_stats()
                
                # Wait for next check
                check_interval = self.config.get('monitoring.check_interval_seconds', 30)
                self.stop_event.wait(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.stop_event.wait(30)  # Wait before retrying
                
        self.logger.info("Monitoring loop ended")
        
    def _is_working_hours(self) -> bool:
        """Check if current time is within configured working hours."""
        try:
            now = datetime.now().time()
            start_time = dt_time.fromisoformat(self.config.get('working_hours.start_time'))
            end_time = dt_time.fromisoformat(self.config.get('working_hours.end_time'))
            
            return start_time <= now <= end_time
            
        except Exception as e:
            self.logger.error(f"Error checking working hours: {e}")
            return True  # Default to always monitoring if there's an error
            
    def _analyze_activity(self) -> None:
        """Analyze current user activity and take appropriate actions."""
        try:
            # Get recent activity features for ML prediction
            features = self.activity_monitor.get_activity_features(window_minutes=10)
            
            if features is None:
                self.logger.debug("Insufficient activity data for analysis")
                return
                
            # Use ML model to classify activity
            prediction, confidence = self.ml_classifier.predict(features)
            
            # Check for anomalous patterns
            is_anomaly = self.ml_classifier.detect_anomaly(features)
            
            self.logger.debug(f"Activity prediction: {prediction} (confidence: {confidence:.3f})")
            
            # Determine if user is currently active
            idle_threshold = self.config.get('monitoring.idle_threshold_minutes', 10)
            is_user_active = self.activity_monitor.is_user_active(idle_threshold)
            
            # Combined decision: ML prediction + basic activity check
            is_working = (prediction == 1 and confidence > self.config.get('model.confidence_threshold', 0.7)) or is_user_active
            
            if is_working:
                self.consecutive_idle_periods = 0
                self.daily_stats['work_minutes'] += self.config.get('monitoring.check_interval_seconds', 30) / 60
            else:
                self.consecutive_idle_periods += 1
                self.daily_stats['idle_minutes'] += self.config.get('monitoring.check_interval_seconds', 30) / 60
                
                # Check if we should send an alert
                self._check_and_send_alert(is_anomaly, features)
                
            # Log activity summary
            self._log_activity_summary(is_working, prediction, confidence, is_anomaly)
            
        except Exception as e:
            self.logger.error(f"Error analyzing activity: {e}")
            
    def _check_and_send_alert(self, is_anomaly: bool, features) -> None:
        """Check if an inactivity alert should be sent."""
        if not self.email_service:
            return
            
        idle_threshold = self.config.get('monitoring.idle_threshold_minutes', 10)
        check_interval_minutes = self.config.get('monitoring.check_interval_seconds', 30) / 60
        
        # Calculate how long user has been idle
        idle_duration_minutes = self.consecutive_idle_periods * check_interval_minutes
        
        if idle_duration_minutes >= idle_threshold:
            # Prepare activity summary for email
            activity_summary = {
                'idle_duration_minutes': idle_duration_minutes,
                'consecutive_idle_periods': self.consecutive_idle_periods,
                'keyboard_events': features[0],
                'mouse_events': features[1],
                'mouse_distance': features[2],
                'screen_change_score': features[3],
                'cpu_usage': features[5],
                'memory_usage': features[6],
                'anomaly_detected': is_anomaly,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Send alert email
            success = self.email_service.send_inactivity_alert(
                idle_duration_minutes=int(idle_duration_minutes),
                activity_summary=activity_summary,
                min_interval_minutes=30  # Don't spam emails
            )
            
            if success:
                self.daily_stats['alerts_sent'] += 1
                self.logger.info(f"Inactivity alert sent (idle for {idle_duration_minutes:.1f} minutes)")
                
    def _update_daily_stats(self) -> None:
        """Update daily activity statistics."""
        recent_activity = self.activity_monitor.get_recent_activity(minutes=1)
        
        if recent_activity:
            latest_activity = recent_activity[-1]
            self.daily_stats['total_keyboard_events'] += latest_activity.keyboard_events
            self.daily_stats['total_mouse_events'] += latest_activity.mouse_events
            
    def _log_activity_summary(self, is_working: bool, prediction: int, confidence: float, is_anomaly: bool) -> None:
        """Log a summary of current activity analysis."""
        status = "WORKING" if is_working else "IDLE"
        anomaly_flag = " [ANOMALY]" if is_anomaly else ""
        
        self.logger.info(f"Activity: {status} | ML: {prediction} ({confidence:.3f}) | "
                        f"Idle periods: {self.consecutive_idle_periods}{anomaly_flag}")
                        
    def _send_daily_summary(self) -> None:
        """Send daily work summary email."""
        if not self.email_service:
            return
            
        try:
            # Calculate summary metrics
            total_minutes = self.daily_stats['work_minutes'] + self.daily_stats['idle_minutes']
            work_hours = self.daily_stats['work_minutes'] / 60
            idle_hours = self.daily_stats['idle_minutes'] / 60
            
            if total_minutes < 30:  # Don't send summary for very short sessions
                return
                
            activity_summary = {
                'session_duration_hours': total_minutes / 60,
                'keyboard_events': self.daily_stats['total_keyboard_events'],
                'mouse_events': self.daily_stats['total_mouse_events'],
                'alerts_sent': self.daily_stats['alerts_sent'],
                'start_time': self.daily_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')
            }
            
            success = self.email_service.send_daily_summary(
                work_hours=work_hours,
                idle_hours=idle_hours,
                activity_summary=activity_summary
            )
            
            if success:
                self.logger.info("Daily summary email sent")
                
        except Exception as e:
            self.logger.error(f"Failed to send daily summary: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the work monitor."""
        return {
            'is_running': self.is_running,
            'in_working_hours': self._is_working_hours(),
            'consecutive_idle_periods': self.consecutive_idle_periods,
            'daily_stats': self.daily_stats.copy(),
            'model_info': self.ml_classifier.get_model_info() if self.ml_classifier else None,
            'email_enabled': self.email_service is not None
        }
        
    def force_retrain_model(self) -> bool:
        """Force retraining of the ML model."""
        try:
            if self.ml_classifier:
                self.logger.info("Force retraining ML model")
                self.ml_classifier.train_model()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to retrain model: {e}")
            return False
            
    def send_test_email(self) -> bool:
        """Send a test email to verify email configuration."""
        if not self.email_service:
            return False
            
        try:
            return self.email_service.test_connection()
        except Exception as e:
            self.logger.error(f"Test email failed: {e}")
            return False
