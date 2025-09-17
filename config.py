"""Configuration settings for AutoDataPipeline."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
DB_DIR = PROJECT_ROOT / "database"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, REPORTS_DIR, DB_DIR]:
    directory.mkdir(exist_ok=True)

# Database configuration
DATABASE_PATH = DB_DIR / "vehicle_data.db"
DATABASE_CONFIG = {
    "path": str(DATABASE_PATH),
    "timeout": 30,
    "check_same_thread": False
}

# Data generation settings
SYNTHETIC_DATA_CONFIG = {
    "num_records": 1000,
    "time_range_hours": 24,
    "speed_range": (0, 120),  # km/h
    "engine_temp_range": (70, 110),  # Celsius
    "fuel_level_range": (0, 100),  # percentage
    "gps_bounds": {
        "lat_min": 40.0,
        "lat_max": 41.0,
        "lon_min": -74.5,
        "lon_max": -73.5
    },
    "anomaly_rate": 0.05  # 5% anomalies
}

# Data transformation settings
TRANSFORMATION_CONFIG = {
    "outlier_threshold": 3.0,  # Standard deviations
    "interpolation_method": "linear",
    "window_size": 10,  # For rolling averages
    "remove_outliers": True,
    "outlier_method": "iqr",
    "normalize_data": True,
    "normalization_method": "standard"
}

# Anomaly detection settings
ANOMALY_DETECTION_CONFIG = {
    "detection_method": "isolation_forest",
    "isolation_forest": {
        "contamination": 0.1,
        "random_state": 42,
        "n_estimators": 100
    },
    "threshold_detection": {
        "speed_max": 130,  # km/h
        "engine_temp_max": 115,  # Celsius
        "fuel_level_min": 5  # percentage
    }
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "output_dir": REPORTS_DIR,
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "colors": {
        "normal": "#2E86AB",
        "anomaly": "#F24236",
        "background": "#F5F5F5",
        "success": "#42B883"
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": LOGS_DIR / "pipeline.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# API configuration (optional)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "AutoDataPipeline API",
    "description": "Vehicle sensor data anomaly detection API",
    "version": "1.0.0"
}

# Export settings
EXPORT_CONFIG = {
    "enable_export": True,
    "format": "parquet",
    "output_dir": REPORTS_DIR,
    "parquet_compression": "snappy",
    "csv_encoding": "utf-8"
}