"""Production-grade structured logging."""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


class StructuredLogger:
    """Structured JSON logger for production environments."""

    def __init__(self, name: str, log_file: str = None, level: int = logging.INFO):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            log_file: Optional file path for logs
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Console handler with JSON formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)

    def log(self, level: str, message: str, **kwargs):
        """Log structured message with context."""
        extra = {'context': kwargs}
        getattr(self.logger, level)(message, extra=extra)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log('info', message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log('warning', message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log('error', message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log('debug', message, **kwargs)


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add custom context
        if hasattr(record, 'context'):
            log_data['context'] = record.context

        return json.dumps(log_data)


# Global logger instance
fraud_logger = StructuredLogger('fraud-detection', log_file='logs/fraud_detection.log')
