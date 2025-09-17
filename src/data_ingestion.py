"""Data ingestion module for AutoDataPipeline."""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config import SYNTHETIC_DATA_CONFIG, DATA_DIR
from .logging_config import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Handles data loading and synthetic data generation."""
    
    def __init__(self):
        """Initialize the data ingestion module."""
        self.config = SYNTHETIC_DATA_CONFIG
        logger.info("DataIngestion module initialized")
    
    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Convert timestamp column if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Successfully loaded {len(df)} records from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise
    
    def load_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            logger.info(f"Loading JSON file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            # Convert timestamp column if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Successfully loaded {len(df)} records from JSON")
            return df
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            raise
    
    def generate_synthetic_data(self, num_records: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic vehicle sensor data.
        
        Args:
            num_records: Number of records to generate (uses config default if None)
            
        Returns:
            DataFrame containing synthetic vehicle sensor data
        """
        num_records = num_records or self.config["num_records"]
        logger.info(f"Generating {num_records} synthetic vehicle sensor records")
        
        # Time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.config["time_range_hours"])
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=start_time,
            end=end_time,
            periods=num_records
        )
        
        # Generate vehicle IDs
        vehicle_ids = [f"VH{random.randint(1000, 9999)}" for _ in range(num_records)]
        
        # Generate normal sensor data
        data = {
            'timestamp': timestamps,
            'vehicle_id': vehicle_ids,
            'speed': self._generate_speed_data(num_records),
            'engine_temperature': self._generate_engine_temp_data(num_records),
            'fuel_level': self._generate_fuel_level_data(num_records),
            'latitude': self._generate_coordinate_data(
                num_records, 
                self.config["gps_bounds"]["lat_min"],
                self.config["gps_bounds"]["lat_max"]
            ),
            'longitude': self._generate_coordinate_data(
                num_records,
                self.config["gps_bounds"]["lon_min"],
                self.config["gps_bounds"]["lon_max"]
            )
        }
        
        df = pd.DataFrame(data)
        
        # Inject anomalies
        df = self._inject_anomalies(df)
        
        logger.info(f"Generated synthetic data with {len(df)} records")
        return df
    
    def _generate_speed_data(self, num_records: int) -> List[float]:
        """Generate realistic speed data."""
        speed_min, speed_max = self.config["speed_range"]
        
        # Generate mostly normal speeds with some variation
        speeds = np.random.normal(60, 20, num_records)  # Mean 60 km/h, std 20
        speeds = np.clip(speeds, speed_min, speed_max)
        
        return speeds.tolist()
    
    def _generate_engine_temp_data(self, num_records: int) -> List[float]:
        """Generate realistic engine temperature data."""
        temp_min, temp_max = self.config["engine_temp_range"]
        
        # Generate mostly normal temperatures
        temps = np.random.normal(90, 8, num_records)  # Mean 90°C, std 8
        temps = np.clip(temps, temp_min, temp_max)
        
        return temps.tolist()
    
    def _generate_fuel_level_data(self, num_records: int) -> List[float]:
        """Generate realistic fuel level data."""
        fuel_min, fuel_max = self.config["fuel_level_range"]
        
        # Generate decreasing fuel levels with some randomness
        base_levels = np.linspace(fuel_max, fuel_min + 20, num_records)
        noise = np.random.normal(0, 5, num_records)
        fuel_levels = base_levels + noise
        fuel_levels = np.clip(fuel_levels, fuel_min, fuel_max)
        
        return fuel_levels.tolist()
    
    def _generate_coordinate_data(self, num_records: int, min_val: float, max_val: float) -> List[float]:
        """Generate GPS coordinate data within bounds."""
        return np.random.uniform(min_val, max_val, num_records).tolist()
    
    def _inject_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject anomalies into the dataset."""
        anomaly_rate = self.config["anomaly_rate"]
        num_anomalies = int(len(df) * anomaly_rate)
        
        if num_anomalies == 0:
            return df
        
        logger.info(f"Injecting {num_anomalies} anomalies ({anomaly_rate*100:.1f}%)")
        
        # Select random indices for anomalies
        anomaly_indices = random.sample(range(len(df)), num_anomalies)
        
        for idx in anomaly_indices:
            anomaly_type = random.choice(['speed', 'temperature', 'fuel'])
            
            if anomaly_type == 'speed':
                # Extremely high speed
                df.loc[idx, 'speed'] = random.uniform(150, 200)
            elif anomaly_type == 'temperature':
                # Overheating
                df.loc[idx, 'engine_temperature'] = random.uniform(120, 140)
            elif anomaly_type == 'fuel':
                # Sudden fuel drop
                df.loc[idx, 'fuel_level'] = random.uniform(0, 5)
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str, format_type: str = 'csv') -> Path:
        """Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            format_type: File format ('csv' or 'json')
            
        Returns:
            Path to the saved file
        """
        output_path = DATA_DIR / filename
        
        try:
            if format_type.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format_type.lower() == 'json':
                df.to_json(output_path, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Data saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {str(e)}")
            raise