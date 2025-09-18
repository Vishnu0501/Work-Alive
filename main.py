#!/usr/bin/env python3
"""
Work Activity Monitor - Main Entry Point

A production-ready Python project that uses machine learning to monitor 
user activity and detect whether the user is actually working or idle.
"""

import sys
import os
from pathlib import Path

# Add the work_monitor package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from work_monitor.cli import cli

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
