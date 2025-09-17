#!/usr/bin/env python3
"""Main entry point for AutoDataPipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import LOGGING_CONFIG
from src.logging_config import setup_logging
from src.pipeline import DataPipeline


def main():
    """Main function to run the AutoDataPipeline."""
    parser = argparse.ArgumentParser(
        description="AutoDataPipeline - Vehicle Sensor Data Processing"
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "process", "full", "api"],
        default="full",
        help="Pipeline execution mode"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input CSV/JSON file path (for process mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--export-parquet",
        action="store_true",
        help="Export results to Parquet format"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else LOGGING_CONFIG["level"]
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        pipeline = DataPipeline()
        
        if args.mode == "generate":
            logger.info("Generating synthetic vehicle sensor data...")
            pipeline.generate_synthetic_data()
            
        elif args.mode == "process":
            if not args.input_file:
                logger.error("Input file required for process mode")
                sys.exit(1)
            logger.info(f"Processing data from {args.input_file}...")
            pipeline.process_file(args.input_file, args.output_dir, args.export_parquet)
            
        elif args.mode == "full":
            logger.info("Running full pipeline (generate + process)...")
            pipeline.run_full_pipeline(args.output_dir, args.export_parquet)
            
        elif args.mode == "api":
            logger.info("Starting FastAPI server...")
            from src.api import start_api_server
            start_api_server()
            
        logger.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()