import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from config import DATABASE_CONFIG
from src.logging_config import get_logger

logger = get_logger(__name__)

class DataStorage:
    """SQLite database storage manager for AutoDataPipeline."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, uses config default.
        """
        self.db_path = db_path or DATABASE_CONFIG['path']
        self.ensure_database_directory()
        self.init_database()
        logger.info(f"Database initialized at: {self.db_path}")
    
    def ensure_database_directory(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.info(f"Created database directory: {db_dir}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database tables based on schema."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create raw_data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS raw_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        vehicle_id TEXT NOT NULL,
                        speed REAL,
                        engine_temp REAL,
                        fuel_level REAL,
                        tire_pressure REAL,
                        battery_voltage REAL,
                        oil_pressure REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for raw_data
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_data(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_vehicle_id ON raw_data(vehicle_id)")
                
                # Create processed_data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processed_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        raw_data_id INTEGER,
                        timestamp DATETIME NOT NULL,
                        vehicle_id TEXT NOT NULL,
                        speed REAL,
                        engine_temp REAL,
                        fuel_level REAL,
                        tire_pressure REAL,
                        battery_voltage REAL,
                        oil_pressure REAL,
                        speed_rolling_mean REAL,
                        engine_temp_rolling_mean REAL,
                        fuel_consumption_rate REAL,
                        speed_change REAL,
                        hour_of_day INTEGER,
                        day_of_week INTEGER,
                        is_weekend INTEGER,
                        processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (raw_data_id) REFERENCES raw_data (id)
                    )
                """)
                
                # Create indexes for processed_data
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_timestamp ON processed_data(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_vehicle_id ON processed_data(vehicle_id)")
                
                # Create anomalies table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS anomalies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        processed_data_id INTEGER,
                        timestamp DATETIME NOT NULL,
                        vehicle_id TEXT NOT NULL,
                        anomaly_score REAL NOT NULL,
                        is_anomaly INTEGER NOT NULL,
                        detection_method TEXT NOT NULL,
                        anomaly_type TEXT,
                        confidence REAL,
                        detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (processed_data_id) REFERENCES processed_data (id)
                    )
                """)
                
                # Create indexes for anomalies
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_vehicle_id ON anomalies(vehicle_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_is_anomaly ON anomalies(is_anomaly)")
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def insert_raw_data(self, data: pd.DataFrame) -> List[int]:
        """Insert raw data into database.
        
        Args:
            data: DataFrame with raw sensor data
            
        Returns:
            List of inserted record IDs
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data for insertion
                records = []
                for _, row in data.iterrows():
                    records.append((
                        str(row['timestamp']),
                        row['vehicle_id'],
                        row.get('speed'),
                        row.get('engine_temp'),
                        row.get('fuel_level'),
                        row.get('tire_pressure'),
                        row.get('battery_voltage'),
                        row.get('oil_pressure')
                    ))
                
                cursor.executemany("""
                    INSERT INTO raw_data 
                    (timestamp, vehicle_id, speed, engine_temp, fuel_level, 
                     tire_pressure, battery_voltage, oil_pressure)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                # Get inserted IDs
                if cursor.lastrowid is not None:
                    first_id = cursor.lastrowid - len(records) + 1
                    inserted_ids = list(range(first_id, cursor.lastrowid + 1))
                else:
                    # Fallback: generate sequential IDs starting from 1
                    inserted_ids = list(range(1, len(records) + 1))
                
                conn.commit()
                logger.info(f"Inserted {len(records)} raw data records")
                return inserted_ids
                
        except Exception as e:
            logger.error(f"Error inserting raw data: {e}")
            raise
    
    def insert_processed_data(self, data: pd.DataFrame, raw_data_ids: List[int]) -> List[int]:
        """Insert processed data into database.
        
        Args:
            data: DataFrame with processed sensor data
            raw_data_ids: List of corresponding raw data IDs
            
        Returns:
            List of inserted record IDs
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data for insertion
                records = []
                for i, (_, row) in enumerate(data.iterrows()):
                    raw_id = raw_data_ids[i] if i < len(raw_data_ids) else None
                    records.append((
                        raw_id,
                        str(row['timestamp']),
                        row['vehicle_id'],
                        row.get('speed'),
                        row.get('engine_temp'),
                        row.get('fuel_level'),
                        row.get('tire_pressure'),
                        row.get('battery_voltage'),
                        row.get('oil_pressure'),
                        row.get('speed_rolling_mean'),
                        row.get('engine_temp_rolling_mean'),
                        row.get('fuel_consumption_rate'),
                        row.get('speed_change'),
                        row.get('hour_of_day'),
                        row.get('day_of_week'),
                        row.get('is_weekend')
                    ))
                
                cursor.executemany("""
                    INSERT INTO processed_data 
                    (raw_data_id, timestamp, vehicle_id, speed, engine_temp, fuel_level,
                     tire_pressure, battery_voltage, oil_pressure, speed_rolling_mean,
                     engine_temp_rolling_mean, fuel_consumption_rate, speed_change,
                     hour_of_day, day_of_week, is_weekend)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                # Get inserted IDs
                if cursor.lastrowid is not None:
                    first_id = cursor.lastrowid - len(records) + 1
                    inserted_ids = list(range(first_id, cursor.lastrowid + 1))
                else:
                    # Fallback: generate sequential IDs starting from 1
                    inserted_ids = list(range(1, len(records) + 1))
                
                conn.commit()
                logger.info(f"Inserted {len(records)} processed data records")
                return inserted_ids
                
        except Exception as e:
            logger.error(f"Error inserting processed data: {e}")
            raise
    
    def insert_anomalies(self, anomalies_data: pd.DataFrame, processed_data_ids: List[int]):
        """Insert anomaly detection results into database.
        
        Args:
            anomalies_data: DataFrame with anomaly detection results
            processed_data_ids: List of corresponding processed data IDs
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data for insertion
                records = []
                for i, (_, row) in enumerate(anomalies_data.iterrows()):
                    processed_id = processed_data_ids[i] if i < len(processed_data_ids) else None
                    
                    # Determine which anomaly column to use
                    is_anomaly = 0
                    if 'anomaly_combined' in row and row['anomaly_combined']:
                        is_anomaly = 1
                    elif 'anomaly_isolation_forest' in row and row['anomaly_isolation_forest']:
                        is_anomaly = 1
                    elif 'anomaly_threshold' in row and row['anomaly_threshold']:
                        is_anomaly = 1
                    
                    records.append((
                        processed_id,
                        str(row['timestamp']),
                        row['vehicle_id'],
                        row.get('isolation_forest_score', 0.0),
                        is_anomaly,
                        row.get('detection_method', 'isolation_forest'),
                        row.get('anomaly_type'),
                        row.get('confidence')
                    ))
                
                cursor.executemany("""
                    INSERT INTO anomalies 
                    (processed_data_id, timestamp, vehicle_id, anomaly_score, is_anomaly,
                     detection_method, anomaly_type, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                conn.commit()
                logger.info(f"Inserted {len(records)} anomaly records")
                
        except Exception as e:
            logger.error(f"Error inserting anomalies: {e}")
            raise
    
    def get_raw_data(self, limit: Optional[int] = None, 
                     vehicle_id: Optional[str] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve raw data from database.
        
        Args:
            limit: Maximum number of records to return
            vehicle_id: Filter by specific vehicle ID
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            
        Returns:
            DataFrame with raw data
        """
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM raw_data WHERE 1=1"
                params = []
                
                if vehicle_id:
                    query += " AND vehicle_id = ?"
                    params.append(vehicle_id)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                return pd.read_sql_query(query, conn, params=params)
                
        except Exception as e:
            logger.error(f"Error retrieving raw data: {e}")
            raise
    
    def get_processed_data(self, limit: Optional[int] = None,
                          vehicle_id: Optional[str] = None) -> pd.DataFrame:
        """Retrieve processed data from database.
        
        Args:
            limit: Maximum number of records to return
            vehicle_id: Filter by specific vehicle ID
            
        Returns:
            DataFrame with processed data
        """
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM processed_data WHERE 1=1"
                params = []
                
                if vehicle_id:
                    query += " AND vehicle_id = ?"
                    params.append(vehicle_id)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                return pd.read_sql_query(query, conn, params=params)
                
        except Exception as e:
            logger.error(f"Error retrieving processed data: {e}")
            raise
    
    def get_anomalies(self, limit: Optional[int] = None,
                     vehicle_id: Optional[str] = None,
                     anomalies_only: bool = True) -> pd.DataFrame:
        """Retrieve anomaly detection results from database.
        
        Args:
            limit: Maximum number of records to return
            vehicle_id: Filter by specific vehicle ID
            anomalies_only: If True, return only detected anomalies
            
        Returns:
            DataFrame with anomaly data
        """
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM anomalies WHERE 1=1"
                params = []
                
                if anomalies_only:
                    query += " AND is_anomaly = 1"
                
                if vehicle_id:
                    query += " AND vehicle_id = ?"
                    params.append(vehicle_id)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                return pd.read_sql_query(query, conn, params=params)
                
        except Exception as e:
            logger.error(f"Error retrieving anomalies: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                cursor.execute("SELECT COUNT(*) FROM raw_data")
                stats['raw_data_count'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM processed_data")
                stats['processed_data_count'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM anomalies")
                stats['total_anomaly_records'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM anomalies WHERE is_anomaly = 1")
                stats['detected_anomalies'] = cursor.fetchone()[0]
                
                # Get unique vehicle count
                cursor.execute("SELECT COUNT(DISTINCT vehicle_id) FROM raw_data")
                stats['unique_vehicles'] = cursor.fetchone()[0]
                
                # Get date range
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM raw_data")
                result = cursor.fetchone()
                stats['data_date_range'] = {
                    'start': result[0],
                    'end': result[1]
                }
                
                logger.info("Retrieved database statistics")
                return stats
                
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            raise
    
    def export_to_parquet(self, output_dir: str):
        """Export all data to Parquet files.
        
        Args:
            output_dir: Directory to save Parquet files
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export each table
            tables = ['raw_data', 'processed_data', 'anomalies']
            
            for table in tables:
                with self.get_connection() as conn:
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    
                    if not df.empty:
                        output_path = os.path.join(output_dir, f"{table}.parquet")
                        df.to_parquet(output_path, index=False)
                        logger.info(f"Exported {table} to {output_path} ({len(df)} records)")
                    else:
                        logger.warning(f"No data found in {table} table")
            
            logger.info(f"Data export completed to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting to Parquet: {e}")
            raise
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from database.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate cutoff date
                cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
                
                # Delete old records (cascading from raw_data)
                cursor.execute("""
                    DELETE FROM anomalies 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                anomalies_deleted = cursor.rowcount
                
                cursor.execute("""
                    DELETE FROM processed_data 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                processed_deleted = cursor.rowcount
                
                cursor.execute("""
                    DELETE FROM raw_data 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                raw_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up old data: {raw_deleted} raw, {processed_deleted} processed, {anomalies_deleted} anomaly records")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            raise