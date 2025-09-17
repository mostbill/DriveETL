"""Data transformation module for AutoDataPipeline."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import TRANSFORMATION_CONFIG
from .logging_config import get_logger

logger = get_logger(__name__)


class DataTransformation:
    """Handles data cleaning, transformation, and feature engineering."""
    
    def __init__(self):
        """Initialize the data transformation module."""
        self.config = TRANSFORMATION_CONFIG
        self.scaler = None
        logger.info("DataTransformation module initialized")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the input data by handling missing values and duplicates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning for {len(df)} records")
        original_count = len(df)
        
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Validate data types
        df_clean = self._validate_data_types(df_clean)
        
        # Remove invalid records
        df_clean = self._remove_invalid_records(df_clean)
        
        cleaned_count = len(df_clean)
        removed_count = original_count - cleaned_count
        
        logger.info(f"Data cleaning completed: {cleaned_count} records remaining, {removed_count} records removed")
        
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'zscore') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove outliers from the dataset.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ('zscore', 'iqr', or 'isolation')
            
        Returns:
            Tuple of (cleaned_data, outliers)
        """
        logger.info(f"Removing outliers using {method} method")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['timestamp']]
        
        if method == 'zscore':
            outlier_mask = self._detect_outliers_zscore(df[numeric_columns])
        elif method == 'iqr':
            outlier_mask = self._detect_outliers_iqr(df[numeric_columns])
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        clean_data = df[~outlier_mask].copy()
        outliers = df[outlier_mask].copy()
        
        logger.info(f"Outlier removal completed: {len(outliers)} outliers detected and removed")
        
        return clean_data, outliers
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")
        
        df_features = df.copy()
        
        # Time-based features
        if 'timestamp' in df_features.columns:
            df_features = self._create_time_features(df_features)
        
        # Rolling statistics
        df_features = self._create_rolling_features(df_features)
        
        # Ratio and derived features
        df_features = self._create_derived_features(df_features)
        
        # Lag features
        df_features = self._create_lag_features(df_features)
        
        logger.info(f"Feature engineering completed: {len(df_features.columns)} total features")
        
        return df_features
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Normalize numerical features.
        
        Args:
            df: Input DataFrame
            method: Normalization method ('standard' or 'minmax')
            
        Returns:
            DataFrame with normalized features
        """
        logger.info(f"Normalizing data using {method} method")
        
        df_normalized = df.copy()
        
        # Select numeric columns (excluding timestamp and IDs)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        exclude_columns = ['timestamp', 'vehicle_id']
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Fit and transform the data
        df_normalized[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        
        logger.info(f"Data normalization completed for {len(numeric_columns)} features")
        
        return df_normalized
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            logger.info(f"Found {total_missing} missing values")
            
            # For numeric columns, use interpolation
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().any():
                    df[col] = df[col].interpolate(method=self.config["interpolation_method"])
            
            # For categorical columns, use forward fill
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(method='ffill')
            
            # Drop rows with remaining missing values
            df = df.dropna()
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} duplicate records")
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure numeric columns are numeric
        numeric_columns = ['speed', 'engine_temperature', 'fuel_level', 'latitude', 'longitude']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with invalid values."""
        initial_count = len(df)
        
        # Remove records with negative speed or fuel level
        if 'speed' in df.columns:
            df = df[df['speed'] >= 0]
        
        if 'fuel_level' in df.columns:
            df = df[df['fuel_level'] >= 0]
        
        # Remove records with unrealistic coordinates
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
        
        final_count = len(df)
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} invalid records")
        
        return df
    
    def _detect_outliers_zscore(self, df: pd.DataFrame) -> pd.Series:
        """Detect outliers using Z-score method."""
        threshold = self.config["outlier_threshold"]
        z_scores = np.abs(stats.zscore(df, nan_policy='omit'))
        return (z_scores > threshold).any(axis=1)
    
    def _detect_outliers_iqr(self, df: pd.DataFrame) -> pd.Series:
        """Detect outliers using IQR method."""
        outlier_mask = pd.Series(False, index=df.index)
        
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            outlier_mask = outlier_mask | column_outliers
        
        return outlier_mask
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'timestamp' not in df.columns:
            return df
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features."""
        window_size = self.config["window_size"]
        numeric_columns = ['speed', 'engine_temperature', 'fuel_level']
        
        for col in numeric_columns:
            if col in df.columns:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones."""
        # Speed categories
        if 'speed' in df.columns:
            df['speed_category'] = pd.cut(
                df['speed'],
                bins=[0, 30, 60, 90, float('inf')],
                labels=['low', 'medium', 'high', 'very_high']
            )
        
        # Engine temperature status
        if 'engine_temperature' in df.columns:
            df['temp_status'] = pd.cut(
                df['engine_temperature'],
                bins=[0, 80, 100, 110, float('inf')],
                labels=['cold', 'normal', 'warm', 'hot']
            )
        
        # Fuel level categories
        if 'fuel_level' in df.columns:
            df['fuel_status'] = pd.cut(
                df['fuel_level'],
                bins=[0, 20, 50, 80, 100],
                labels=['critical', 'low', 'medium', 'high']
            )
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series analysis."""
        numeric_columns = ['speed', 'engine_temperature', 'fuel_level']
        
        for col in numeric_columns:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_diff'] = df[col] - df[f'{col}_lag1']
        
        return df