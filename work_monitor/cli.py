"""
Command Line Interface Module

Provides CLI commands for controlling the work activity monitor.
"""

import click
import time
import signal
import sys
from datetime import datetime, time as dt_time
from typing import Optional
from pathlib import Path

from .config import ConfigManager
from .logging_setup import setup_logging, get_logger
from .activity_monitor import ActivityMonitor
from .ml_model import ActivityClassifier
from .email_service import EmailService
from .work_monitor import WorkMonitor


@click.group()
@click.option('--config', '-c', default='config.yaml', 
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Work Activity Monitor - ML-based productivity monitoring."""
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        config_manager = ConfigManager(config)
        ctx.obj['config'] = config_manager
        
        # Setup logging
        log_level = 'DEBUG' if verbose else config_manager.get('logging.level', 'INFO')
        log_file = config_manager.get_log_file_path()
        max_size = config_manager.get('logging.max_log_size_mb', 50)
        backup_count = config_manager.get('logging.backup_count', 5)
        
        setup_logging(str(log_file), log_level, max_size, backup_count)
        ctx.obj['logger'] = get_logger('cli')
        
        # Ensure directories exist
        config_manager.ensure_directories()
        
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--start-time', '-s', default=None,
              help='Working start time (HH:MM format, e.g., 09:00)')
@click.option('--end-time', '-e', default=None,
              help='Working end time (HH:MM format, e.g., 17:00)')
@click.option('--idle-threshold', '-t', type=int, default=None,
              help='Idle threshold in minutes before alert')
@click.option('--check-interval', '-i', type=int, default=None,
              help='Activity check interval in seconds')
@click.pass_context
def start(ctx, start_time, end_time, idle_threshold, check_interval):
    """Start monitoring user activity."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    logger.info("Starting Work Activity Monitor")
    
    # Update configuration with command line arguments
    if start_time:
        config.set('working_hours.start_time', start_time)
    if end_time:
        config.set('working_hours.end_time', end_time)
    if idle_threshold:
        config.set('monitoring.idle_threshold_minutes', idle_threshold)
    if check_interval:
        config.set('monitoring.check_interval_seconds', check_interval)
    
    # Validate configuration
    if not config.validate_working_hours():
        click.echo("Error: Invalid working hours configuration", err=True)
        sys.exit(1)
        
    if not config.validate_email_config():
        click.echo("Warning: Email configuration invalid - notifications disabled")
    
    try:
        # Initialize work monitor
        work_monitor = WorkMonitor(config)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            work_monitor.stop()
            click.echo("\nWork monitor stopped.")
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start monitoring
        work_monitor.start()
        
        click.echo("Work Activity Monitor started successfully!")
        click.echo(f"Working hours: {config.get('working_hours.start_time')} - {config.get('working_hours.end_time')}")
        click.echo(f"Idle threshold: {config.get('monitoring.idle_threshold_minutes')} minutes")
        click.echo("Press Ctrl+C to stop monitoring...")
        
        # Keep the main thread alive
        try:
            while work_monitor.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            work_monitor.stop()
            
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        click.echo(f"Error starting monitor: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stop(ctx):
    """Stop the work activity monitor."""
    logger = ctx.obj['logger']
    logger.info("Stop command received")
    click.echo("Work monitor stopped.")


@cli.command()
@click.option('--retrain', is_flag=True, help='Force retrain the ML model')
@click.option('--samples', type=int, default=1000, 
              help='Number of synthetic samples for training')
@click.pass_context
def train(ctx, retrain, samples):
    """Train or retrain the ML model."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    logger.info("Training ML model")
    click.echo("Training ML model for activity classification...")
    
    try:
        # Initialize model
        model_path = config.get_model_path()
        classifier = ActivityClassifier(str(model_path))
        
        # Check if retraining is needed
        if not retrain and classifier.is_model_trained():
            if not classifier.retrain_if_needed():
                click.echo("Model is already trained and up to date.")
                return
        
        # Train model
        click.echo(f"Generating {samples} synthetic training samples...")
        metrics = classifier.train_model(use_synthetic=True)
        
        # Display results
        click.echo("\nTraining Results:")
        click.echo(f"  Test Accuracy: {metrics['test_accuracy']:.3f}")
        click.echo(f"  CV Accuracy: {metrics['cv_mean']:.3f} (±{metrics['cv_std']:.3f})")
        
        click.echo("\nTop Features:")
        for feature, importance in sorted(metrics['feature_importance'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5]:
            click.echo(f"  {feature}: {importance:.3f}")
            
        click.echo(f"\nModel saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"Error training model: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def test_email(ctx):
    """Test email configuration by sending a test email."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    logger.info("Testing email configuration")
    
    if not config.validate_email_config():
        click.echo("Error: Email configuration is invalid", err=True)
        sys.exit(1)
        
    try:
        # Initialize email service
        email_service = EmailService(
            smtp_server=config.get('email.smtp_server'),
            smtp_port=config.get('email.smtp_port'),
            sender_email=config.get('email.sender_email'),
            sender_password=config.get('email.sender_password'),
            recipient_email=config.get('email.recipient_email')
        )
        
        click.echo("Sending test email...")
        success = email_service.test_connection()
        
        if success:
            click.echo("✓ Test email sent successfully!")
            click.echo(f"Email sent to: {config.get('email.recipient_email')}")
        else:
            click.echo("✗ Failed to send test email", err=True)
            click.echo("Please check your email configuration and credentials")
            
    except Exception as e:
        logger.error(f"Email test failed: {e}")
        click.echo(f"Error testing email: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--minutes', type=int, default=60,
              help='Number of minutes of activity data to show')
@click.pass_context
def status(ctx, minutes):
    """Show current activity status and recent statistics."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    try:
        # Initialize activity monitor for status check
        activity_monitor = ActivityMonitor(
            check_interval=config.get('monitoring.check_interval_seconds', 30),
            screenshot_interval=config.get('monitoring.screenshot_interval_seconds', 60)
        )
        
        # Check if we have recent activity data
        recent_activity = activity_monitor.get_recent_activity(minutes)
        
        click.echo(f"Work Activity Monitor Status")
        click.echo("=" * 30)
        click.echo(f"Configuration file: {config.config_path}")
        click.echo(f"Working hours: {config.get('working_hours.start_time')} - {config.get('working_hours.end_time')}")
        click.echo(f"Idle threshold: {config.get('monitoring.idle_threshold_minutes')} minutes")
        
        # Check if currently in working hours
        now = datetime.now().time()
        start_time = dt_time.fromisoformat(config.get('working_hours.start_time'))
        end_time = dt_time.fromisoformat(config.get('working_hours.end_time'))
        
        in_working_hours = start_time <= now <= end_time
        click.echo(f"Currently in working hours: {'Yes' if in_working_hours else 'No'}")
        
        # Model status
        model_path = config.get_model_path()
        if model_path.exists():
            classifier = ActivityClassifier(str(model_path))
            if classifier.is_model_trained():
                model_info = classifier.get_model_info()
                click.echo(f"ML Model: Trained ({model_info.get('accuracy', 'Unknown'):.3f} accuracy)")
            else:
                click.echo("ML Model: Not trained")
        else:
            click.echo("ML Model: Not found")
            
        # Email configuration
        email_valid = config.validate_email_config()
        click.echo(f"Email notifications: {'Configured' if email_valid else 'Not configured'}")
        
        if recent_activity:
            click.echo(f"\nRecent Activity ({len(recent_activity)} records in last {minutes} minutes):")
            total_keyboard = sum(a.keyboard_events for a in recent_activity)
            total_mouse = sum(a.mouse_events for a in recent_activity)
            total_distance = sum(a.mouse_distance for a in recent_activity)
            
            click.echo(f"  Keyboard events: {total_keyboard}")
            click.echo(f"  Mouse events: {total_mouse}")
            click.echo(f"  Mouse distance: {total_distance:.1f} pixels")
        else:
            click.echo(f"\nNo recent activity data available")
            
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        click.echo(f"Error checking status: {e}", err=True)


@cli.command()
@click.option('--output', '-o', default='activity_data.csv',
              help='Output file for exported data')
@click.pass_context
def export(ctx, output):
    """Export activity data to CSV file."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    try:
        # Initialize activity monitor
        activity_monitor = ActivityMonitor()
        
        # Export data
        activity_monitor.export_activity_data(output)
        click.echo(f"Activity data exported to: {output}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        click.echo(f"Error exporting data: {e}", err=True)


@cli.command()
@click.option('--key', required=True, help='Configuration key (e.g., email.sender_email)')
@click.option('--value', required=True, help='Configuration value')
@click.pass_context
def config_set(ctx, key, value):
    """Set a configuration value."""
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    try:
        # Convert value to appropriate type
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
            
        config.set(key, value)
        config.save_config()
        
        click.echo(f"Configuration updated: {key} = {value}")
        logger.info(f"Configuration updated: {key} = {value}")
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        click.echo(f"Error updating configuration: {e}", err=True)


@cli.command()
@click.option('--key', help='Specific configuration key to show')
@click.pass_context
def config_get(ctx, key):
    """Get configuration value(s)."""
    config = ctx.obj['config']
    
    if key:
        value = config.get(key)
        if value is not None:
            click.echo(f"{key}: {value}")
        else:
            click.echo(f"Configuration key '{key}' not found")
    else:
        # Show all configuration
        import yaml
        click.echo(yaml.dump(config.config, default_flow_style=False))


if __name__ == '__main__':
    cli()
