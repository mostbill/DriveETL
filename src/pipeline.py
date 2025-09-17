import pandas as pd
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from config import (
    SYNTHETIC_DATA_CONFIG, TRANSFORMATION_CONFIG, 
    ANOMALY_DETECTION_CONFIG, EXPORT_CONFIG
)
from src.logging_config import get_logger
from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.anomaly_detection import AnomalyDetector
from src.data_storage import DataStorage
from src.visualization import DataVisualizer

logger = get_logger(__name__)

class DataPipeline:
    """Main data pipeline orchestrator for AutoDataPipeline."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.ingestion = DataIngestion()
        self.transformation = DataTransformation()
        self.anomaly_detector = AnomalyDetector()
        self.storage = DataStorage()
        self.visualizer = DataVisualizer()
        
        logger.info("Data pipeline initialized successfully")
    
    def generate_sample_data(self, num_records: Optional[int] = None) -> pd.DataFrame:
        """Generate sample sensor data.
        
        Args:
            num_records: Number of records to generate. If None, uses config default.
            
        Returns:
            DataFrame with generated sensor data
        """
        try:
            num_records = num_records or SYNTHETIC_DATA_CONFIG['num_records']
            logger.info(f"Generating {num_records} sample records")
            
            data = self.ingestion.generate_synthetic_data(
                num_records=num_records
            )
            
            logger.info(f"Generated {len(data)} records for {data['vehicle_id'].nunique()} vehicles")
            return data
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            raise
    
    def load_data_from_file(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV or JSON file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading data from file: {file_path}")
            
            if file_path.lower().endswith('.csv'):
                data = self.ingestion.load_csv(file_path)
            elif file_path.lower().endswith('.json'):
                data = self.ingestion.load_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from file: {e}")
            raise
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw data through transformation pipeline.
        
        Args:
            data: Raw sensor data
            
        Returns:
            Processed and cleaned data
        """
        try:
            logger.info(f"Processing {len(data)} records")
            
            # Data cleaning
            cleaned_data = self.transformation.clean_data(data)
            logger.info(f"Data cleaning completed: {len(cleaned_data)} records remaining")
            
            # Outlier removal
            if TRANSFORMATION_CONFIG['remove_outliers']:
                cleaned_data, outliers = self.transformation.remove_outliers(
                    cleaned_data, 
                    method=TRANSFORMATION_CONFIG['outlier_method']
                )
                logger.info(f"Outlier removal completed: {len(cleaned_data)} records remaining")
            
            # Feature engineering
            processed_data = self.transformation.engineer_features(cleaned_data)
            logger.info(f"Feature engineering completed: {processed_data.shape[1]} features")
            
            # Data normalization
            if TRANSFORMATION_CONFIG['normalize_data']:
                processed_data = self.transformation.normalize_data(
                    processed_data,
                    method=TRANSFORMATION_CONFIG['normalization_method']
                )
                logger.info("Data normalization completed")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in processed data.
        
        Args:
            data: Processed sensor data
            
        Returns:
            DataFrame with anomaly detection results
        """
        try:
            logger.info(f"Detecting anomalies in {len(data)} records")
            
            # Train anomaly detection model
            self.anomaly_detector.train_isolation_forest(data)
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(
                data, 
                method=ANOMALY_DETECTION_CONFIG['detection_method']
            )
            
            # Log statistics
            anomaly_column = 'anomaly_combined' if ANOMALY_DETECTION_CONFIG['detection_method'] == 'combined' else f"anomaly_{ANOMALY_DETECTION_CONFIG['detection_method']}"
            total_anomalies = len(anomalies[anomalies[anomaly_column] == True])
            anomaly_rate = (total_anomalies / len(anomalies)) * 100 if len(anomalies) > 0 else 0
            
            logger.info(f"Anomaly detection completed: {total_anomalies} anomalies detected ({anomaly_rate:.2f}% rate)")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise
    
    def store_data(self, raw_data: pd.DataFrame, 
                   processed_data: pd.DataFrame, 
                   anomalies: pd.DataFrame) -> Dict[str, List[int]]:
        """Store data in database.
        
        Args:
            raw_data: Raw sensor data
            processed_data: Processed sensor data
            anomalies: Anomaly detection results
            
        Returns:
            Dictionary with inserted record IDs
        """
        try:
            logger.info("Storing data in database")
            
            # Store raw data
            raw_ids = self.storage.insert_raw_data(raw_data)
            
            # Store processed data
            processed_ids = self.storage.insert_processed_data(processed_data, raw_ids)
            
            # Store anomalies
            self.storage.insert_anomalies(anomalies, processed_ids)
            
            logger.info("Data storage completed successfully")
            
            return {
                'raw_ids': raw_ids,
                'processed_ids': processed_ids
            }
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise
    
    def create_visualizations(self, data: pd.DataFrame, 
                            anomalies: pd.DataFrame) -> List[str]:
        """Create visualization dashboard.
        
        Args:
            data: Processed sensor data
            anomalies: Anomaly detection results
            
        Returns:
            List of paths to saved visualization files
        """
        try:
            logger.info("Creating visualization dashboard")
            
            # Create comprehensive dashboard
            plot_files = self.visualizer.create_dashboard(data, anomalies)
            
            logger.info(f"Visualization dashboard created: {len(plot_files)} plots generated")
            return plot_files
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise
    
    def export_data(self, export_format: str = 'parquet'):
        """Export data to external formats.
        
        Args:
            export_format: Export format ('parquet' or 'csv')
        """
        try:
            if not EXPORT_CONFIG['enable_export']:
                logger.info("Data export is disabled in configuration")
                return
            
            logger.info(f"Exporting data in {export_format} format")
            
            if export_format.lower() == 'parquet':
                self.storage.export_to_parquet(EXPORT_CONFIG['output_dir'])
            else:
                logger.warning(f"Export format '{export_format}' not implemented")
            
            logger.info("Data export completed")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise
    
    def generate_report(self, data: pd.DataFrame, 
                       anomalies: pd.DataFrame) -> str:
        """Generate analysis summary report.
        
        Args:
            data: Processed sensor data
            anomalies: Anomaly detection results
            
        Returns:
            Path to generated report file
        """
        try:
            logger.info("Generating analysis summary report")
            
            # Get database statistics
            stats = self.storage.get_statistics()
            
            # Generate report
            report_path = self.visualizer.generate_summary_report(data, anomalies, stats)
            
            logger.info(f"Analysis report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def run_full_pipeline(self, input_file: Optional[str] = None, 
                         num_records: Optional[int] = None) -> Dict[str, Any]:
        """Execute the complete data pipeline.
        
        Args:
            input_file: Path to input data file (optional)
            num_records: Number of records to generate if no input file (optional)
            
        Returns:
            Dictionary with pipeline execution results
        """
        try:
            start_time = datetime.now()
            logger.info("Starting full pipeline execution")
            
            # Step 1: Data Ingestion
            if input_file and os.path.exists(input_file):
                raw_data = self.load_data_from_file(input_file)
                data_source = f"file: {input_file}"
            else:
                raw_data = self.generate_sample_data(num_records)
                data_source = "synthetic data generation"
            
            logger.info(f"Data ingestion completed from {data_source}")
            
            # Step 2: Data Processing
            processed_data = self.process_data(raw_data)
            
            # Step 3: Anomaly Detection
            anomalies = self.detect_anomalies(processed_data)
            
            # Step 4: Data Storage
            storage_ids = self.store_data(raw_data, processed_data, anomalies)
            
            # Step 5: Visualization
            plot_files = self.create_visualizations(processed_data, anomalies)
            
            # Step 6: Report Generation
            report_path = self.generate_report(processed_data, anomalies)
            
            # Step 7: Data Export (optional)
            if EXPORT_CONFIG['enable_export']:
                self.export_data(EXPORT_CONFIG['format'])
            
            # Calculate execution time
            execution_time = datetime.now() - start_time
            
            # Compile results
            anomaly_column = 'anomaly_combined' if ANOMALY_DETECTION_CONFIG['detection_method'] == 'combined' else f"anomaly_{ANOMALY_DETECTION_CONFIG['detection_method']}"
            anomalies_count = len(anomalies[anomalies[anomaly_column] == True])
            results = {
                'execution_time': str(execution_time),
                'data_source': data_source,
                'records_processed': len(raw_data),
                'anomalies_detected': anomalies_count,
                'anomaly_rate': (anomalies_count / len(anomalies)) * 100 if len(anomalies) > 0 else 0,
                'storage_ids': storage_ids,
                'visualization_files': plot_files,
                'report_file': report_path,
                'database_stats': self.storage.get_statistics()
            }
            
            logger.info(f"Full pipeline execution completed in {execution_time}")
            logger.info(f"Results: {results['records_processed']} records processed, {results['anomalies_detected']} anomalies detected")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in full pipeline execution: {e}")
            raise
    
    def run_data_generation_only(self, num_records: Optional[int] = None, 
                                output_file: Optional[str] = None) -> str:
        """Generate and save sample data only.
        
        Args:
            num_records: Number of records to generate
            output_file: Path to save generated data
            
        Returns:
            Path to saved data file
        """
        try:
            logger.info("Running data generation mode")
            
            # Generate data
            data = self.generate_sample_data(num_records)
            
            # Save data
            if not output_file:
                output_file = f"data/generated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            self.ingestion.save_to_csv(data, output_file)
            
            logger.info(f"Sample data generated and saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error in data generation: {e}")
            raise
    
    def run_processing_only(self, input_file: str) -> Dict[str, Any]:
        """Process existing data file without full pipeline.
        
        Args:
            input_file: Path to input data file
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Running processing mode for file: {input_file}")
            
            # Load and process data
            raw_data = self.load_data_from_file(input_file)
            processed_data = self.process_data(raw_data)
            anomalies = self.detect_anomalies(processed_data)
            
            # Create visualizations
            plot_files = self.create_visualizations(processed_data, anomalies)
            
            # Generate report
            report_path = self.generate_report(processed_data, anomalies)
            
            anomaly_column = 'anomaly_combined' if ANOMALY_DETECTION_CONFIG['detection_method'] == 'combined' else f"anomaly_{ANOMALY_DETECTION_CONFIG['detection_method']}"
            anomalies_count = len(anomalies[anomalies[anomaly_column] == True])
            results = {
                'input_file': input_file,
                'records_processed': len(raw_data),
                'anomalies_detected': anomalies_count,
                'anomaly_rate': (anomalies_count / len(anomalies)) * 100 if len(anomalies) > 0 else 0,
                'visualization_files': plot_files,
                'report_file': report_path
            }
            
            logger.info(f"Processing completed: {results['records_processed']} records, {results['anomalies_detected']} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"Error in processing mode: {e}")
            raise
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from database.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        try:
            logger.info(f"Cleaning up data older than {days_to_keep} days")
            self.storage.cleanup_old_data(days_to_keep)
            logger.info("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in data cleanup: {e}")
            raise