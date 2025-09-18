"""
Email Notification Service

Handles sending email alerts when user inactivity is detected.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from .logging_setup import get_logger


class EmailService:
    """Service for sending email notifications."""
    
    def __init__(self, 
                 smtp_server: str,
                 smtp_port: int,
                 sender_email: str,
                 sender_password: str,
                 recipient_email: str):
        """
        Initialize the email service.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender email password (app password recommended)
            recipient_email: Recipient email address
        """
        self.logger = get_logger(__name__)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        
        # Email tracking
        self.last_notification_time: Optional[datetime] = None
        self.notification_count = 0
        
    def send_inactivity_alert(self, 
                            idle_duration_minutes: int,
                            activity_summary: Optional[Dict[str, Any]] = None,
                            min_interval_minutes: int = 30) -> bool:
        """
        Send an inactivity alert email.
        
        Args:
            idle_duration_minutes: How long the user has been idle
            activity_summary: Summary of recent activity data
            min_interval_minutes: Minimum interval between notifications
            
        Returns:
            True if email sent successfully, False otherwise
        """
        # Check if we should send notification (rate limiting)
        if not self._should_send_notification(min_interval_minutes):
            self.logger.debug("Skipping notification due to rate limiting")
            return False
            
        try:
            # Create email message
            message = self._create_inactivity_message(idle_duration_minutes, activity_summary)
            
            # Send email
            success = self._send_email(message)
            
            if success:
                self.last_notification_time = datetime.now()
                self.notification_count += 1
                self.logger.info(f"Inactivity alert sent (idle for {idle_duration_minutes} minutes)")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send inactivity alert: {e}")
            return False
            
    def send_daily_summary(self, 
                          work_hours: float,
                          idle_hours: float,
                          activity_summary: Dict[str, Any]) -> bool:
        """
        Send a daily work summary email.
        
        Args:
            work_hours: Total hours worked
            idle_hours: Total hours idle
            activity_summary: Summary of daily activity
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            message = self._create_daily_summary_message(work_hours, idle_hours, activity_summary)
            success = self._send_email(message)
            
            if success:
                self.logger.info("Daily summary email sent")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send daily summary: {e}")
            return False
            
    def send_system_alert(self, 
                         alert_type: str,
                         message: str,
                         details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a system alert email.
        
        Args:
            alert_type: Type of alert (ERROR, WARNING, INFO)
            message: Alert message
            details: Additional details dictionary
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            email_message = self._create_system_alert_message(alert_type, message, details)
            success = self._send_email(email_message)
            
            if success:
                self.logger.info(f"System alert sent: {alert_type}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send system alert: {e}")
            return False
            
    def test_connection(self) -> bool:
        """
        Test the email connection and configuration.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create test message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = self.recipient_email
            message["Subject"] = "Work Monitor - Email Test"
            
            body = """
This is a test email from the Work Activity Monitor.

If you receive this email, your email configuration is working correctly.

Test Details:
- Timestamp: {}
- SMTP Server: {}
- Port: {}
- Sender: {}
- Recipient: {}

Best regards,
Work Activity Monitor
            """.format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.smtp_server,
                self.smtp_port,
                self.sender_email,
                self.recipient_email
            )
            
            message.attach(MIMEText(body, "plain"))
            
            # Send test email
            success = self._send_email(message)
            
            if success:
                self.logger.info("Email test successful")
            else:
                self.logger.error("Email test failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Email test failed: {e}")
            return False
            
    def _should_send_notification(self, min_interval_minutes: int) -> bool:
        """Check if enough time has passed since last notification."""
        if self.last_notification_time is None:
            return True
            
        time_since_last = datetime.now() - self.last_notification_time
        return time_since_last.total_seconds() >= (min_interval_minutes * 60)
        
    def _create_inactivity_message(self, 
                                  idle_duration_minutes: int,
                                  activity_summary: Optional[Dict[str, Any]]) -> MIMEMultipart:
        """Create inactivity alert email message."""
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.recipient_email
        message["Subject"] = f"Work Monitor Alert - User Inactive for {idle_duration_minutes} minutes"
        
        # Create email body
        body_parts = []
        body_parts.append("WORK ACTIVITY MONITOR ALERT")
        body_parts.append("=" * 40)
        body_parts.append("")
        body_parts.append(f"User has been inactive for {idle_duration_minutes} minutes.")
        body_parts.append(f"Alert triggered at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        body_parts.append("")
        
        # Add activity summary if available
        if activity_summary:
            body_parts.append("Recent Activity Summary:")
            body_parts.append("-" * 25)
            for key, value in activity_summary.items():
                if isinstance(value, float):
                    body_parts.append(f"{key}: {value:.2f}")
                else:
                    body_parts.append(f"{key}: {value}")
            body_parts.append("")
            
        body_parts.append("Please check if the user is actually working or if there are any issues.")
        body_parts.append("")
        body_parts.append("This is an automated message from the Work Activity Monitor system.")
        body_parts.append(f"Notification #{self.notification_count + 1}")
        
        body = "\n".join(body_parts)
        message.attach(MIMEText(body, "plain"))
        
        return message
        
    def _create_daily_summary_message(self,
                                    work_hours: float,
                                    idle_hours: float,
                                    activity_summary: Dict[str, Any]) -> MIMEMultipart:
        """Create daily summary email message."""
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.recipient_email
        message["Subject"] = f"Work Monitor Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Calculate productivity metrics
        total_hours = work_hours + idle_hours
        productivity_percent = (work_hours / total_hours * 100) if total_hours > 0 else 0
        
        body_parts = []
        body_parts.append("DAILY WORK ACTIVITY SUMMARY")
        body_parts.append("=" * 40)
        body_parts.append("")
        body_parts.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        body_parts.append(f"Total Hours Monitored: {total_hours:.1f}")
        body_parts.append(f"Active Work Time: {work_hours:.1f} hours ({productivity_percent:.1f}%)")
        body_parts.append(f"Idle Time: {idle_hours:.1f} hours")
        body_parts.append("")
        
        # Add detailed activity metrics
        body_parts.append("Activity Metrics:")
        body_parts.append("-" * 18)
        for key, value in activity_summary.items():
            if isinstance(value, float):
                body_parts.append(f"{key}: {value:.2f}")
            else:
                body_parts.append(f"{key}: {value}")
        body_parts.append("")
        
        body_parts.append("This is an automated daily summary from the Work Activity Monitor system.")
        
        body = "\n".join(body_parts)
        message.attach(MIMEText(body, "plain"))
        
        return message
        
    def _create_system_alert_message(self,
                                   alert_type: str,
                                   alert_message: str,
                                   details: Optional[Dict[str, Any]]) -> MIMEMultipart:
        """Create system alert email message."""
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.recipient_email
        message["Subject"] = f"Work Monitor System Alert - {alert_type}"
        
        body_parts = []
        body_parts.append(f"WORK MONITOR SYSTEM ALERT - {alert_type}")
        body_parts.append("=" * 50)
        body_parts.append("")
        body_parts.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        body_parts.append(f"Alert Type: {alert_type}")
        body_parts.append("")
        body_parts.append("Message:")
        body_parts.append(alert_message)
        body_parts.append("")
        
        if details:
            body_parts.append("Additional Details:")
            body_parts.append("-" * 20)
            for key, value in details.items():
                body_parts.append(f"{key}: {value}")
            body_parts.append("")
            
        body_parts.append("Please check the Work Activity Monitor system for more information.")
        body_parts.append("")
        body_parts.append("This is an automated system alert.")
        
        body = "\n".join(body_parts)
        message.attach(MIMEText(body, "plain"))
        
        return message
        
    def _send_email(self, message: MIMEMultipart) -> bool:
        """
        Send email message via SMTP.
        
        Args:
            message: Email message to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect to server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                
                text = message.as_string()
                server.sendmail(self.sender_email, self.recipient_email, text)
                
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPRecipientsRefused as e:
            self.logger.error(f"SMTP recipients refused: {e}")
            return False
        except smtplib.SMTPServerDisconnected as e:
            self.logger.error(f"SMTP server disconnected: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
            
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get email notification statistics."""
        return {
            'total_notifications': self.notification_count,
            'last_notification': self.last_notification_time,
            'smtp_server': self.smtp_server,
            'sender_email': self.sender_email,
            'recipient_email': self.recipient_email
        }
