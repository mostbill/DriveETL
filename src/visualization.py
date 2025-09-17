import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from config import VISUALIZATION_CONFIG
from src.logging_config import get_logger

logger = get_logger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataVisualizer:
    """Data visualization manager for AutoDataPipeline."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots. If None, uses config default.
        """
        self.output_dir = output_dir or VISUALIZATION_CONFIG['output_dir']
        self.figure_size = VISUALIZATION_CONFIG['figure_size']
        self.dpi = VISUALIZATION_CONFIG['dpi']
        self.ensure_output_directory()
        logger.info(f"Visualizer initialized with output directory: {self.output_dir}")
    
    def ensure_output_directory(self):
        """Ensure the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def plot_sensor_data_overview(self, data: pd.DataFrame, 
                                 anomalies: Optional[pd.DataFrame] = None,
                                 save_plot: bool = True) -> str:
        """Create overview plots of all sensor data.
        
        Args:
            data: DataFrame with sensor data
            anomalies: DataFrame with anomaly data (optional)
            save_plot: Whether to save the plot to file
            
        Returns:
            Path to saved plot file
        """
        try:
            # Prepare data
            data = data.copy()
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp')
            
            # Define sensor columns
            sensor_cols = ['speed', 'engine_temp', 'fuel_level', 'tire_pressure', 
                          'battery_voltage', 'oil_pressure']
            available_cols = [col for col in sensor_cols if col in data.columns]
            
            if not available_cols:
                logger.warning("No sensor columns found in data")
                return ""
            
            # Create subplots
            n_cols = 2
            n_rows = (len(available_cols) + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figure_size[0] * 1.5, 
                                                             self.figure_size[1] * n_rows * 0.6))
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle('Vehicle Sensor Data Overview', fontsize=16, fontweight='bold')
            
            # Plot each sensor
            for i, col in enumerate(available_cols):
                row, col_idx = divmod(i, n_cols)
                ax = axes[row, col_idx]
                
                # Plot normal data
                if 'timestamp' in data.columns:
                    ax.plot(data['timestamp'], data[col], alpha=0.7, linewidth=1, 
                           label='Normal Data', color='blue')
                    
                    # Highlight anomalies if provided
                    if anomalies is not None and not anomalies.empty:
                        # Determine which anomaly column to use
                        anomaly_col = None
                        if 'anomaly_combined' in anomalies.columns:
                            anomaly_col = 'anomaly_combined'
                        elif 'anomaly_isolation_forest' in anomalies.columns:
                            anomaly_col = 'anomaly_isolation_forest'
                        elif 'anomaly_threshold' in anomalies.columns:
                            anomaly_col = 'anomaly_threshold'
                        elif 'is_anomaly' in anomalies.columns:
                            anomaly_col = 'is_anomaly'
                        
                        if anomaly_col:
                            anomaly_data = anomalies[anomalies[anomaly_col] == 1]
                        else:
                            anomaly_data = pd.DataFrame()
                        if not anomaly_data.empty and 'timestamp' in anomaly_data.columns:
                            anomaly_data['timestamp'] = pd.to_datetime(anomaly_data['timestamp'])
                            # Match anomalies with original data
                            merged = pd.merge(anomaly_data[['timestamp', 'vehicle_id']], 
                                            data, on=['timestamp', 'vehicle_id'], how='inner')
                            if not merged.empty and col in merged.columns:
                                ax.scatter(merged['timestamp'], merged[col], 
                                         color='red', s=30, alpha=0.8, 
                                         label='Anomalies', zorder=5)
                else:
                    ax.plot(data.index, data[col], alpha=0.7, linewidth=1, 
                           label='Normal Data', color='blue')
                    
                    if anomalies is not None and not anomalies.empty:
                        # Determine which anomaly column to use
                        anomaly_col = None
                        if 'anomaly_combined' in anomalies.columns:
                            anomaly_col = 'anomaly_combined'
                        elif 'anomaly_isolation_forest' in anomalies.columns:
                            anomaly_col = 'anomaly_isolation_forest'
                        elif 'anomaly_threshold' in anomalies.columns:
                            anomaly_col = 'anomaly_threshold'
                        elif 'is_anomaly' in anomalies.columns:
                            anomaly_col = 'is_anomaly'
                        
                        if anomaly_col:
                            anomaly_indices = anomalies[anomalies[anomaly_col] == 1].index
                        else:
                            anomaly_indices = []
                        if len(anomaly_indices) > 0 and col in data.columns:
                            ax.scatter(anomaly_indices, data.loc[anomaly_indices, col], 
                                     color='red', s=30, alpha=0.8, 
                                     label='Anomalies', zorder=5)
                
                ax.set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Format x-axis for time series
                if 'timestamp' in data.columns:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Hide empty subplots
            for i in range(len(available_cols), n_rows * n_cols):
                row, col_idx = divmod(i, n_cols)
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            
            if save_plot:
                filename = f"sensor_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Sensor overview plot saved: {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error creating sensor overview plot: {e}")
            raise
    
    def plot_anomaly_distribution(self, anomalies: pd.DataFrame, 
                                 save_plot: bool = True) -> str:
        """Create anomaly distribution plots.
        
        Args:
            anomalies: DataFrame with anomaly detection results
            save_plot: Whether to save the plot to file
            
        Returns:
            Path to saved plot file
        """
        try:
            if anomalies.empty:
                logger.warning("No anomaly data provided")
                return ""
            
            fig, axes = plt.subplots(2, 2, figsize=(self.figure_size[0] * 1.5, 
                                                    self.figure_size[1] * 1.2))
            fig.suptitle('Anomaly Detection Analysis', fontsize=16, fontweight='bold')
            
            # 1. Anomaly score distribution
            ax1 = axes[0, 0]
            if 'anomaly_score' in anomalies.columns:
                ax1.hist(anomalies['anomaly_score'], bins=30, alpha=0.7, 
                        color='skyblue', edgecolor='black')
                ax1.axvline(anomalies['anomaly_score'].mean(), color='red', 
                           linestyle='--', label=f"Mean: {anomalies['anomaly_score'].mean():.3f}")
                ax1.set_title('Anomaly Score Distribution')
                ax1.set_xlabel('Anomaly Score')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Anomalies over time
            ax2 = axes[0, 1]
            if 'timestamp' in anomalies.columns:
                anomalies_copy = anomalies.copy()
                anomalies_copy['timestamp'] = pd.to_datetime(anomalies_copy['timestamp'])
                anomalies_copy = anomalies_copy.sort_values('timestamp')
                
                # Group by hour and count anomalies
                anomalies_copy['hour'] = anomalies_copy['timestamp'].dt.floor('H')
                
                # Determine which anomaly column to use
                anomaly_col = None
                if 'anomaly_combined' in anomalies_copy.columns:
                    anomaly_col = 'anomaly_combined'
                elif 'anomaly_isolation_forest' in anomalies_copy.columns:
                    anomaly_col = 'anomaly_isolation_forest'
                elif 'anomaly_threshold' in anomalies_copy.columns:
                    anomaly_col = 'anomaly_threshold'
                elif 'is_anomaly' in anomalies_copy.columns:
                    anomaly_col = 'is_anomaly'
                
                if anomaly_col:
                    hourly_anomalies = anomalies_copy[anomalies_copy[anomaly_col] == 1].groupby('hour').size()
                else:
                    hourly_anomalies = pd.Series(dtype=int)
                
                if not hourly_anomalies.empty:
                    ax2.plot(hourly_anomalies.index, hourly_anomalies.values, 
                            marker='o', linewidth=2, markersize=4, color='red')
                    ax2.set_title('Anomalies Over Time')
                    ax2.set_xlabel('Time')
                    ax2.set_ylabel('Number of Anomalies')
                    ax2.grid(True, alpha=0.3)
                    
                    # Format x-axis
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # 3. Anomalies by vehicle
            ax3 = axes[1, 0]
            if 'vehicle_id' in anomalies.columns:
                # Determine which anomaly column to use
                anomaly_col = None
                if 'anomaly_combined' in anomalies.columns:
                    anomaly_col = 'anomaly_combined'
                elif 'anomaly_isolation_forest' in anomalies.columns:
                    anomaly_col = 'anomaly_isolation_forest'
                elif 'anomaly_threshold' in anomalies.columns:
                    anomaly_col = 'anomaly_threshold'
                elif 'is_anomaly' in anomalies.columns:
                    anomaly_col = 'is_anomaly'
                
                if anomaly_col:
                    vehicle_anomalies = anomalies[anomalies[anomaly_col] == 1]['vehicle_id'].value_counts()
                else:
                    vehicle_anomalies = pd.Series(dtype=int)
                if not vehicle_anomalies.empty:
                    vehicle_anomalies.plot(kind='bar', ax=ax3, color='orange', alpha=0.7)
                    ax3.set_title('Anomalies by Vehicle')
                    ax3.set_xlabel('Vehicle ID')
                    ax3.set_ylabel('Number of Anomalies')
                    ax3.tick_params(axis='x', rotation=45)
                    ax3.grid(True, alpha=0.3)
            
            # 4. Detection method comparison
            ax4 = axes[1, 1]
            if 'detection_method' in anomalies.columns:
                # Determine which anomaly column to use
                anomaly_col = None
                if 'anomaly_combined' in anomalies.columns:
                    anomaly_col = 'anomaly_combined'
                elif 'anomaly_isolation_forest' in anomalies.columns:
                    anomaly_col = 'anomaly_isolation_forest'
                elif 'anomaly_threshold' in anomalies.columns:
                    anomaly_col = 'anomaly_threshold'
                elif 'is_anomaly' in anomalies.columns:
                    anomaly_col = 'is_anomaly'
                
                if anomaly_col:
                    method_counts = anomalies[anomalies[anomaly_col] == 1]['detection_method'].value_counts()
                else:
                    method_counts = pd.Series(dtype=int)
                if not method_counts.empty:
                    colors = plt.cm.Set3(np.linspace(0, 1, len(method_counts)))
                    wedges, texts, autotexts = ax4.pie(method_counts.values, 
                                                      labels=method_counts.index,
                                                      autopct='%1.1f%%',
                                                      colors=colors)
                    ax4.set_title('Anomalies by Detection Method')
            
            plt.tight_layout()
            
            if save_plot:
                filename = f"anomaly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Anomaly analysis plot saved: {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error creating anomaly distribution plot: {e}")
            raise
    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                               save_plot: bool = True) -> str:
        """Create correlation matrix heatmap.
        
        Args:
            data: DataFrame with sensor data
            save_plot: Whether to save the plot to file
            
        Returns:
            Path to saved plot file
        """
        try:
            # Select numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                logger.warning("Not enough numeric columns for correlation matrix")
                return ""
            
            # Calculate correlation matrix
            corr_matrix = data[numeric_cols].corr()
            
            # Create heatmap
            plt.figure(figsize=self.figure_size)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, 
                       cbar_kws={"shrink": .8}, fmt='.2f')
            
            plt.title('Sensor Data Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_plot:
                filename = f"correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Correlation matrix plot saved: {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            raise
    
    def plot_vehicle_comparison(self, data: pd.DataFrame, 
                               anomalies: Optional[pd.DataFrame] = None,
                               metric: str = 'speed',
                               save_plot: bool = True) -> str:
        """Create vehicle comparison plots.
        
        Args:
            data: DataFrame with sensor data
            anomalies: DataFrame with anomaly data (optional)
            metric: Sensor metric to compare
            save_plot: Whether to save the plot to file
            
        Returns:
            Path to saved plot file
        """
        try:
            if metric not in data.columns:
                logger.warning(f"Metric '{metric}' not found in data")
                return ""
            
            if 'vehicle_id' not in data.columns:
                logger.warning("vehicle_id column not found in data")
                return ""
            
            fig, axes = plt.subplots(2, 1, figsize=(self.figure_size[0], 
                                                    self.figure_size[1] * 1.5))
            fig.suptitle(f'Vehicle Comparison - {metric.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # 1. Box plot comparison
            ax1 = axes[0]
            vehicles = sorted(data['vehicle_id'].unique())
            vehicle_data = [data[data['vehicle_id'] == v][metric].dropna() for v in vehicles]
            
            bp = ax1.boxplot(vehicle_data, labels=vehicles, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(vehicles)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax1.set_title(f'{metric.replace("_", " ").title()} Distribution by Vehicle')
            ax1.set_xlabel('Vehicle ID')
            ax1.set_ylabel(metric.replace('_', ' ').title())
            ax1.grid(True, alpha=0.3)
            
            # 2. Time series comparison
            ax2 = axes[1]
            if 'timestamp' in data.columns:
                data_copy = data.copy()
                data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
                
                for i, vehicle in enumerate(vehicles[:5]):  # Limit to 5 vehicles for readability
                    vehicle_data = data_copy[data_copy['vehicle_id'] == vehicle].sort_values('timestamp')
                    if not vehicle_data.empty:
                        ax2.plot(vehicle_data['timestamp'], vehicle_data[metric], 
                                label=f'Vehicle {vehicle}', alpha=0.7, linewidth=1.5)
                
                # Highlight anomalies if provided
                if anomalies is not None and not anomalies.empty:
                    # Determine anomaly column
                    anomaly_col = None
                    for col in ['anomaly_combined', 'anomaly_isolation_forest', 'anomaly_threshold', 'is_anomaly']:
                        if col in anomalies.columns:
                            anomaly_col = col
                            break
                    
                    if anomaly_col:
                        anomaly_data = anomalies[anomalies[anomaly_col] == 1]
                        if not anomaly_data.empty and 'timestamp' in anomaly_data.columns:
                            anomaly_data['timestamp'] = pd.to_datetime(anomaly_data['timestamp'])
                        merged = pd.merge(anomaly_data[['timestamp', 'vehicle_id']], 
                                        data_copy, on=['timestamp', 'vehicle_id'], how='inner')
                        if not merged.empty:
                            ax2.scatter(merged['timestamp'], merged[metric], 
                                      color='red', s=20, alpha=0.8, 
                                      label='Anomalies', zorder=5)
                
                ax2.set_title(f'{metric.replace("_", " ").title()} Over Time')
                ax2.set_xlabel('Time')
                ax2.set_ylabel(metric.replace('_', ' ').title())
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
                
                # Format x-axis
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_plot:
                filename = f"vehicle_comparison_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Vehicle comparison plot saved: {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error creating vehicle comparison plot: {e}")
            raise
    
    def create_dashboard(self, data: pd.DataFrame, 
                        anomalies: pd.DataFrame,
                        save_plot: bool = True) -> List[str]:
        """Create a comprehensive dashboard with all visualizations.
        
        Args:
            data: DataFrame with sensor data
            anomalies: DataFrame with anomaly detection results
            save_plot: Whether to save plots to files
            
        Returns:
            List of paths to saved plot files
        """
        try:
            saved_plots = []
            
            logger.info("Creating comprehensive visualization dashboard")
            
            # 1. Sensor data overview
            plot_path = self.plot_sensor_data_overview(data, anomalies, save_plot)
            if plot_path:
                saved_plots.append(plot_path)
            
            # 2. Anomaly analysis
            if not anomalies.empty:
                plot_path = self.plot_anomaly_distribution(anomalies, save_plot)
                if plot_path:
                    saved_plots.append(plot_path)
            
            # 3. Correlation matrix
            plot_path = self.plot_correlation_matrix(data, save_plot)
            if plot_path:
                saved_plots.append(plot_path)
            
            # 4. Vehicle comparisons for key metrics
            key_metrics = ['speed', 'engine_temp', 'fuel_level']
            for metric in key_metrics:
                if metric in data.columns:
                    plot_path = self.plot_vehicle_comparison(data, anomalies, metric, save_plot)
                    if plot_path:
                        saved_plots.append(plot_path)
            
            logger.info(f"Dashboard created with {len(saved_plots)} plots")
            return saved_plots
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    def generate_summary_report(self, data: pd.DataFrame, 
                               anomalies: pd.DataFrame,
                               stats: Dict[str, Any]) -> str:
        """Generate a text summary report.
        
        Args:
            data: DataFrame with sensor data
            anomalies: DataFrame with anomaly detection results
            stats: Database statistics
            
        Returns:
            Path to saved report file
        """
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("AutoDataPipeline - Analysis Summary Report")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Data overview
            report_lines.append("DATA OVERVIEW:")
            report_lines.append("-" * 20)
            report_lines.append(f"Total records processed: {len(data):,}")
            report_lines.append(f"Unique vehicles: {stats.get('unique_vehicles', 'N/A')}")
            report_lines.append(f"Date range: {stats.get('data_date_range', {}).get('start', 'N/A')} to {stats.get('data_date_range', {}).get('end', 'N/A')}")
            report_lines.append("")
            
            # Anomaly summary
            if not anomalies.empty:
                # Determine anomaly column
                anomaly_col = None
                for col in ['anomaly_combined', 'anomaly_isolation_forest', 'anomaly_threshold', 'is_anomaly']:
                    if col in anomalies.columns:
                        anomaly_col = col
                        break
                
                if anomaly_col:
                    total_anomalies = len(anomalies[anomalies[anomaly_col] == 1])
                    anomaly_rate = (total_anomalies / len(data)) * 100 if len(data) > 0 else 0
                else:
                    total_anomalies = 0
                    anomaly_rate = 0
                
                report_lines.append("ANOMALY DETECTION RESULTS:")
                report_lines.append("-" * 30)
                report_lines.append(f"Total anomalies detected: {total_anomalies:,}")
                report_lines.append(f"Anomaly rate: {anomaly_rate:.2f}%")
                
                if 'anomaly_score' in anomalies.columns:
                    avg_score = anomalies['anomaly_score'].mean()
                    max_score = anomalies['anomaly_score'].max()
                    report_lines.append(f"Average anomaly score: {avg_score:.3f}")
                    report_lines.append(f"Maximum anomaly score: {max_score:.3f}")
                
                if 'vehicle_id' in anomalies.columns and anomaly_col:
                    vehicle_anomalies = anomalies[anomalies[anomaly_col] == 1]['vehicle_id'].value_counts()
                    if not vehicle_anomalies.empty:
                        report_lines.append(f"Most affected vehicle: {vehicle_anomalies.index[0]} ({vehicle_anomalies.iloc[0]} anomalies)")
                
                report_lines.append("")
            
            # Sensor statistics
            sensor_cols = ['speed', 'engine_temp', 'fuel_level', 'tire_pressure', 
                          'battery_voltage', 'oil_pressure']
            available_sensors = [col for col in sensor_cols if col in data.columns]
            
            if available_sensors:
                report_lines.append("SENSOR STATISTICS:")
                report_lines.append("-" * 20)
                for sensor in available_sensors:
                    sensor_data = data[sensor].dropna()
                    if not sensor_data.empty:
                        report_lines.append(f"{sensor.replace('_', ' ').title()}:")
                        report_lines.append(f"  Mean: {sensor_data.mean():.2f}")
                        report_lines.append(f"  Std: {sensor_data.std():.2f}")
                        report_lines.append(f"  Min: {sensor_data.min():.2f}")
                        report_lines.append(f"  Max: {sensor_data.max():.2f}")
                        report_lines.append("")
            
            # Save report
            report_content = "\n".join(report_lines)
            filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Summary report saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            raise