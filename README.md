# Work Activity Monitor

A production-ready Python project that uses machine learning to monitor user activity on a computer screen and detect whether the user is actually working or idle.

## Features

- **ML-based Activity Detection**: Uses machine learning to distinguish between working and idle states
- **Customizable Working Hours**: Set your expected working hours via CLI
- **Email Notifications**: Automatic alerts when idle time exceeds thresholds
- **Screen Activity Analysis**: Monitors screen changes, keyboard, and mouse activity
- **Production-Ready**: Modular architecture, comprehensive logging, error handling
- **Extensible Design**: Easy to replace ML models and add new features

## Project Structure

```
Work Alive/
├── work_monitor/           # Main package
│   ├── __init__.py
│   ├── activity_monitor.py # Activity monitoring and data collection
│   ├── ml_model.py         # Machine learning classification
│   ├── email_service.py    # Email notification service
│   ├── config.py           # Configuration management
│   ├── logging_setup.py    # Logging configuration
│   ├── cli.py              # Command line interface
│   └── work_monitor.py     # Main orchestration class
├── models/                 # ML model storage
├── logs/                   # Log files
├── data/                   # Data storage
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── main.py                 # Entry point
└── README.md              # This file
```

## Installation

1. **Clone or download the project**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the system** (edit `config.yaml`):
   - Set your email credentials for notifications
   - Adjust working hours and monitoring thresholds
   - Configure ML model parameters

## Configuration

Edit `config.yaml` to customize the system:

```yaml
# Email settings (required for notifications)
email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "your_email@gmail.com"
  sender_password: "your_app_password"  # Use app password for Gmail
  recipient_email: "manager@company.com"

# Monitoring settings
monitoring:
  idle_threshold_minutes: 10      # Alert after 10 minutes of inactivity
  check_interval_seconds: 30      # Check activity every 30 seconds
  screenshot_interval_seconds: 60 # Take screenshots every minute

# Working hours (24-hour format)
working_hours:
  start_time: "09:00"
  end_time: "17:00"
  timezone: "UTC"

# ML Model settings
model:
  confidence_threshold: 0.7       # Minimum confidence for predictions
  retrain_interval_days: 7        # Retrain model every 7 days
```

## Usage

### Command Line Interface

The system provides a comprehensive CLI for all operations:

```bash
# Show all available commands
python main.py --help

# Start monitoring with default settings
python main.py start

# Start monitoring with custom working hours
python main.py start --start-time 08:00 --end-time 18:00 --idle-threshold 15

# Train the ML model
python main.py train

# Test email configuration
python main.py test-email

# Check system status
python main.py status

# Export activity data to CSV
python main.py export --output my_activity.csv

# Set configuration values
python main.py config-set --key email.sender_email --value your@email.com

# Get configuration values
python main.py config-get --key working_hours.start_time
```

### Basic Workflow

1. **Initial Setup**:
   ```bash
   # Configure email settings
   python main.py config-set --key email.sender_email --value your@email.com
   python main.py config-set --key email.sender_password --value your_app_password
   python main.py config-set --key email.recipient_email --value manager@company.com
   
   # Test email configuration
   python main.py test-email
   
   # Train the ML model
   python main.py train
   ```

2. **Start Monitoring**:
   ```bash
   # Start with default 9 AM - 5 PM working hours
   python main.py start
   
   # Or customize working hours
   python main.py start --start-time 08:30 --end-time 17:30
   ```

3. **Monitor Status**:
   ```bash
   # Check current status
   python main.py status
   
   # Export data for analysis
   python main.py export --output weekly_report.csv
   ```

## How It Works

### Activity Detection

The system monitors multiple activity indicators:

- **Keyboard Events**: Counts key presses and releases
- **Mouse Activity**: Tracks mouse movement, clicks, and scrolling
- **Screen Changes**: Analyzes screenshot differences to detect content changes
- **System Resources**: Monitors CPU and memory usage patterns

### Machine Learning Classification

The ML model uses a Random Forest classifier trained on activity features:

- **Feature Engineering**: Extracts meaningful patterns from raw activity data
- **Synthetic Training Data**: Generates realistic training samples for initial model
- **Continuous Learning**: Model retrains periodically with new data
- **Anomaly Detection**: Identifies unusual activity patterns

### Smart Notifications

Email alerts are sent intelligently:

- **Rate Limiting**: Prevents spam by limiting notification frequency
- **Context-Aware**: Includes detailed activity summaries in alerts
- **Working Hours Only**: Only monitors during configured work hours
- **Escalation**: Tracks consecutive idle periods

## Email Setup

### Gmail Configuration

1. Enable 2-factor authentication on your Gmail account
2. Generate an app password:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. Use the app password in `config.yaml`

### Other Email Providers

Update the SMTP settings in `config.yaml`:

```yaml
email:
  smtp_server: "your-smtp-server.com"
  smtp_port: 587  # or 465 for SSL
  # ... other settings
```

## Logging

The system provides comprehensive logging:

- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **File Rotation**: Automatic log file rotation when size limits reached
- **Console Output**: Real-time status updates
- **Structured Logging**: Detailed context for debugging

Log files are stored in the `logs/` directory.

## Security Considerations

- **Credentials**: Store email passwords securely (use app passwords)
- **Privacy**: Screenshots are processed locally and not stored permanently
- **Permissions**: The system requires permissions to monitor keyboard/mouse input
- **Network**: Only outbound SMTP connections for email notifications

## Troubleshooting

### Common Issues

1. **Email Not Working**:
   ```bash
   python main.py test-email
   ```
   Check SMTP settings and credentials.

2. **Model Not Training**:
   ```bash
   python main.py train --retrain --samples 2000
   ```
   Force retrain with more samples.

3. **Permission Errors**:
   - Run as administrator on Windows
   - Grant accessibility permissions on macOS
   - Install required system packages on Linux

4. **High CPU Usage**:
   - Increase `check_interval_seconds` in config
   - Reduce `screenshot_interval_seconds`

### Debug Mode

Enable verbose logging:

```bash
python main.py --verbose start
```

### Log Analysis

Check log files for detailed information:

```bash
# View recent logs
tail -f logs/activity_monitor.log

# Search for errors
grep ERROR logs/activity_monitor.log
```

## Development

### Adding New Features

The modular architecture makes it easy to extend:

1. **New Activity Sources**: Add to `activity_monitor.py`
2. **ML Models**: Replace or extend `ml_model.py`
3. **Notification Channels**: Add alongside `email_service.py`
4. **CLI Commands**: Extend `cli.py`

### Testing

```bash
# Install development dependencies
pip install pytest black flake8

# Run tests (if implemented)
pytest

# Code formatting
black work_monitor/

# Linting
flake8 work_monitor/
```

## Performance

### System Requirements

- **RAM**: 200-500 MB depending on activity history size
- **CPU**: Low impact, periodic checks every 30 seconds
- **Storage**: ~10 MB for models and logs
- **Network**: Minimal (only for email notifications)

### Optimization Tips

- Adjust `check_interval_seconds` based on needs
- Reduce `screenshot_interval_seconds` for better performance
- Limit activity history size in configuration

## License

This project is provided as-is for educational and personal use.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review log files for error details
3. Verify configuration settings
4. Test individual components (email, model training, etc.)

## Version History

- **v1.0.0**: Initial release with ML-based activity detection, email notifications, and CLI interface
