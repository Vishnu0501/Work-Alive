"""
Configuration Management Module

Handles loading and managing configuration settings from YAML files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages application configuration settings."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
                
            # Validate required sections
            required_sections = ['email', 'monitoring', 'working_hours', 'model', 'logging']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required configuration section: {section}")
                    
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'email.smtp_server')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'email.smtp_server')
            value: Value to set
        """
        keys = key.split('.')
        config_ref = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
            
        # Set the final value
        config_ref[keys[-1]] = value
        
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            raise
            
    def validate_email_config(self) -> bool:
        """Validate email configuration settings."""
        required_fields = [
            'email.smtp_server',
            'email.smtp_port',
            'email.sender_email',
            'email.sender_password',
            'email.recipient_email'
        ]
        
        for field in required_fields:
            if not self.get(field):
                logging.error(f"Missing email configuration: {field}")
                return False
                
        return True
        
    def validate_working_hours(self) -> bool:
        """Validate working hours configuration."""
        start_time = self.get('working_hours.start_time')
        end_time = self.get('working_hours.end_time')
        
        if not start_time or not end_time:
            logging.error("Missing working hours configuration")
            return False
            
        # Validate time format (HH:MM)
        import re
        time_pattern = r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$'
        
        if not re.match(time_pattern, start_time):
            logging.error(f"Invalid start time format: {start_time}")
            return False
            
        if not re.match(time_pattern, end_time):
            logging.error(f"Invalid end time format: {end_time}")
            return False
            
        return True
        
    def get_model_path(self) -> Path:
        """Get the full path to the ML model file."""
        model_path = self.get('model.model_path', 'models/activity_classifier.joblib')
        return Path(model_path)
        
    def get_log_file_path(self) -> Path:
        """Get the full path to the log file."""
        log_file = self.get('logging.log_file', 'logs/activity_monitor.log')
        return Path(log_file)
        
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.get_model_path().parent,
            self.get_log_file_path().parent,
            Path('data')
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
