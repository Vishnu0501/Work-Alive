"""
Activity Monitoring Module

Captures and analyzes user activity including screen changes, 
keyboard input, and mouse movement.
"""

import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import psutil
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    
from PIL import ImageGrab

try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    
import logging

from .logging_setup import get_logger


@dataclass
class ActivityData:
    """Data structure to store activity metrics."""
    timestamp: datetime
    keyboard_events: int = 0
    mouse_events: int = 0
    mouse_distance: float = 0.0
    screen_change_score: float = 0.0
    active_window_changes: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert activity data to feature vector for ML model."""
        return np.array([
            self.keyboard_events,
            self.mouse_events,
            self.mouse_distance,
            self.screen_change_score,
            self.active_window_changes,
            self.cpu_usage,
            self.memory_usage
        ])


class ActivityMonitor:
    """Monitors user activity on the computer."""
    
    def __init__(self, 
                 check_interval: int = 30,
                 screenshot_interval: int = 60,
                 history_size: int = 100):
        """
        Initialize the activity monitor.
        
        Args:
            check_interval: Seconds between activity checks
            screenshot_interval: Seconds between screenshots for screen analysis
            history_size: Number of activity records to keep in memory
        """
        self.logger = get_logger(__name__)
        self.check_interval = check_interval
        self.screenshot_interval = screenshot_interval
        
        # Activity tracking
        self.activity_history: deque = deque(maxlen=history_size)
        self.current_activity = ActivityData(timestamp=datetime.now())
        
        # Input tracking
        self.keyboard_events = 0
        self.mouse_events = 0
        self.mouse_positions: List[Tuple[int, int]] = []
        self.last_mouse_pos: Optional[Tuple[int, int]] = None
        
        # Screen monitoring
        self.last_screenshot: Optional[np.ndarray] = None
        self.screenshot_thread: Optional[threading.Thread] = None
        
        # System monitoring
        self.process = psutil.Process()
        
        # Control flags
        self.is_monitoring = False
        self.stop_event = threading.Event()
        
        # Input listeners
        self.keyboard_listener: Optional[keyboard.Listener] = None
        self.mouse_listener: Optional[mouse.Listener] = None
        
    def start_monitoring(self) -> None:
        """Start monitoring user activity."""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already active")
            return
            
        self.logger.info("Starting activity monitoring")
        self.is_monitoring = True
        self.stop_event.clear()
        
        # Start input listeners
        self._start_input_listeners()
        
        # Start screenshot monitoring thread
        self.screenshot_thread = threading.Thread(target=self._screenshot_monitor, daemon=True)
        self.screenshot_thread.start()
        
        # Start main monitoring loop
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop monitoring user activity."""
        if not self.is_monitoring:
            return
            
        self.logger.info("Stopping activity monitoring")
        self.is_monitoring = False
        self.stop_event.set()
        
        # Stop input listeners
        self._stop_input_listeners()
        
        # Wait for threads to finish
        if self.screenshot_thread and self.screenshot_thread.is_alive():
            self.screenshot_thread.join(timeout=5)
            
    def _start_input_listeners(self) -> None:
        """Start keyboard and mouse input listeners."""
        if not PYNPUT_AVAILABLE:
            self.logger.warning("pynput not available - input monitoring disabled")
            return
            
        try:
            # Keyboard listener
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.keyboard_listener.start()
            
            # Mouse listener
            self.mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll
            )
            self.mouse_listener.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start input listeners: {e}")
            
    def _stop_input_listeners(self) -> None:
        """Stop keyboard and mouse input listeners."""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            
        if self.mouse_listener:
            self.mouse_listener.stop()
            
    def _on_key_press(self, key) -> None:
        """Handle keyboard press events."""
        if self.is_monitoring:
            self.keyboard_events += 1
            
    def _on_key_release(self, key) -> None:
        """Handle keyboard release events."""
        pass  # We only count press events to avoid double counting
        
    def _on_mouse_move(self, x: int, y: int) -> None:
        """Handle mouse movement events."""
        if self.is_monitoring:
            if self.last_mouse_pos:
                # Calculate distance moved
                distance = np.sqrt((x - self.last_mouse_pos[0])**2 + 
                                 (y - self.last_mouse_pos[1])**2)
                if distance > 5:  # Ignore very small movements (noise)
                    self.mouse_positions.append((x, y))
                    self.mouse_events += 1
                    
            self.last_mouse_pos = (x, y)
            
    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        """Handle mouse click events."""
        if self.is_monitoring and pressed:
            self.mouse_events += 1
            
    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        """Handle mouse scroll events."""
        if self.is_monitoring:
            self.mouse_events += 1
            
    def _screenshot_monitor(self) -> None:
        """Monitor screen changes in a separate thread."""
        while self.is_monitoring and not self.stop_event.is_set():
            try:
                # Take screenshot
                screenshot = ImageGrab.grab()
                screenshot_np = np.array(screenshot)
                
                # Calculate screen change score
                if self.last_screenshot is not None:
                    change_score = self._calculate_screen_change(
                        self.last_screenshot, screenshot_np
                    )
                    self.current_activity.screen_change_score = change_score
                    
                self.last_screenshot = screenshot_np
                
            except Exception as e:
                self.logger.error(f"Screenshot monitoring error: {e}")
                
            # Wait for next screenshot
            self.stop_event.wait(self.screenshot_interval)
            
    def _calculate_screen_change(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate screen change score between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Change score (0-1, higher means more change)
        """
        try:
            if not CV2_AVAILABLE:
                # Fallback to simple pixel difference if OpenCV not available
                diff = np.abs(img1.astype(float) - img2.astype(float))
                change_score = np.mean(diff) / 255.0
                return min(change_score, 1.0)
                
            # Resize images for faster processing
            height, width = 100, 100
            img1_small = cv2.resize(img1, (width, height))
            img2_small = cv2.resize(img2, (width, height))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1_small, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2_small, cv2.COLOR_RGB2GRAY)
            
            # Calculate structural similarity
            diff = cv2.absdiff(gray1, gray2)
            change_score = np.mean(diff) / 255.0
            
            return min(change_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating screen change: {e}")
            return 0.0
            
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects activity data."""
        while self.is_monitoring and not self.stop_event.is_set():
            try:
                # Collect activity data
                self._collect_activity_data()
                
                # Reset counters for next interval
                self._reset_activity_counters()
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                
            # Wait for next check
            self.stop_event.wait(self.check_interval)
            
    def _collect_activity_data(self) -> None:
        """Collect current activity data and add to history."""
        try:
            # Calculate mouse movement distance
            mouse_distance = 0.0
            if len(self.mouse_positions) > 1:
                for i in range(1, len(self.mouse_positions)):
                    prev_pos = self.mouse_positions[i-1]
                    curr_pos = self.mouse_positions[i]
                    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                     (curr_pos[1] - prev_pos[1])**2)
                    mouse_distance += distance
                    
            # Get system resource usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            # Update current activity data
            self.current_activity.timestamp = datetime.now()
            self.current_activity.keyboard_events = self.keyboard_events
            self.current_activity.mouse_events = self.mouse_events
            self.current_activity.mouse_distance = mouse_distance
            self.current_activity.cpu_usage = cpu_usage
            self.current_activity.memory_usage = memory_info.percent
            
            # Add to history
            self.activity_history.append(ActivityData(
                timestamp=self.current_activity.timestamp,
                keyboard_events=self.current_activity.keyboard_events,
                mouse_events=self.current_activity.mouse_events,
                mouse_distance=self.current_activity.mouse_distance,
                screen_change_score=self.current_activity.screen_change_score,
                active_window_changes=self.current_activity.active_window_changes,
                cpu_usage=self.current_activity.cpu_usage,
                memory_usage=self.current_activity.memory_usage
            ))
            
            self.logger.debug(f"Activity data collected: {self.current_activity}")
            
        except Exception as e:
            self.logger.error(f"Error collecting activity data: {e}")
            
    def _reset_activity_counters(self) -> None:
        """Reset activity counters for next monitoring interval."""
        self.keyboard_events = 0
        self.mouse_events = 0
        self.mouse_positions.clear()
        self.current_activity.screen_change_score = 0.0
        self.current_activity.active_window_changes = 0
        
    def get_recent_activity(self, minutes: int = 10) -> List[ActivityData]:
        """
        Get recent activity data within specified time window.
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            List of activity data within time window
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            activity for activity in self.activity_history 
            if activity.timestamp >= cutoff_time
        ]
        
    def is_user_active(self, threshold_minutes: int = 5) -> bool:
        """
        Check if user has been active recently.
        
        Args:
            threshold_minutes: Minutes to check for activity
            
        Returns:
            True if user has been active, False otherwise
        """
        recent_activity = self.get_recent_activity(threshold_minutes)
        
        if not recent_activity:
            return False
            
        # Check for any significant activity
        total_keyboard = sum(a.keyboard_events for a in recent_activity)
        total_mouse = sum(a.mouse_events for a in recent_activity)
        total_mouse_distance = sum(a.mouse_distance for a in recent_activity)
        avg_screen_change = np.mean([a.screen_change_score for a in recent_activity])
        
        # Activity thresholds
        keyboard_threshold = 10  # minimum key presses
        mouse_threshold = 20     # minimum mouse events
        mouse_distance_threshold = 500  # minimum mouse movement pixels
        screen_change_threshold = 0.1   # minimum screen change score
        
        is_active = (
            total_keyboard >= keyboard_threshold or
            total_mouse >= mouse_threshold or
            total_mouse_distance >= mouse_distance_threshold or
            avg_screen_change >= screen_change_threshold
        )
        
        self.logger.debug(f"Activity check - Keyboard: {total_keyboard}, "
                         f"Mouse: {total_mouse}, Distance: {total_mouse_distance:.1f}, "
                         f"Screen: {avg_screen_change:.3f}, Active: {is_active}")
        
        return is_active
        
    def get_activity_features(self, window_minutes: int = 10) -> Optional[np.ndarray]:
        """
        Get activity features for ML model prediction.
        
        Args:
            window_minutes: Time window for feature extraction
            
        Returns:
            Feature vector or None if insufficient data
        """
        recent_activity = self.get_recent_activity(window_minutes)
        
        if not recent_activity:
            return None
            
        # Aggregate features over time window
        total_keyboard = sum(a.keyboard_events for a in recent_activity)
        total_mouse = sum(a.mouse_events for a in recent_activity)
        total_mouse_distance = sum(a.mouse_distance for a in recent_activity)
        avg_screen_change = np.mean([a.screen_change_score for a in recent_activity])
        total_window_changes = sum(a.active_window_changes for a in recent_activity)
        avg_cpu_usage = np.mean([a.cpu_usage for a in recent_activity])
        avg_memory_usage = np.mean([a.memory_usage for a in recent_activity])
        
        return np.array([
            total_keyboard,
            total_mouse,
            total_mouse_distance,
            avg_screen_change,
            total_window_changes,
            avg_cpu_usage,
            avg_memory_usage
        ])
        
    def export_activity_data(self, filepath: str) -> None:
        """
        Export activity history to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        import pandas as pd
        
        try:
            data = []
            for activity in self.activity_history:
                data.append({
                    'timestamp': activity.timestamp,
                    'keyboard_events': activity.keyboard_events,
                    'mouse_events': activity.mouse_events,
                    'mouse_distance': activity.mouse_distance,
                    'screen_change_score': activity.screen_change_score,
                    'active_window_changes': activity.active_window_changes,
                    'cpu_usage': activity.cpu_usage,
                    'memory_usage': activity.memory_usage
                })
                
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Activity data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export activity data: {e}")
            raise
