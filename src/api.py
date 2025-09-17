from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
import tempfile
from datetime import datetime
import uvicorn

from config import API_CONFIG
from src.logging_config import get_logger
from src.pipeline import DataPipeline

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AutoDataPipeline API",
    description="REST API for vehicle sensor data anomaly detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize pipeline
pipeline = DataPipeline()

# Pydantic models for request/response
class SensorDataPoint(BaseModel):
    """Single sensor data point model."""
    vehicle_id: str = Field(..., description="Vehicle identifier")
    timestamp: str = Field(..., description="Timestamp in ISO format")
    speed: float = Field(..., ge=0, le=200, description="Vehicle speed in km/h")
    engine_temp: float = Field(..., ge=0, le=150, description="Engine temperature in Celsius")
    fuel_level: float = Field(..., ge=0, le=100, description="Fuel level percentage")
    tire_pressure: float = Field(..., ge=0, le=50, description="Tire pressure in PSI")
    battery_voltage: float = Field(..., ge=0, le=15, description="Battery voltage")
    oil_pressure: float = Field(..., ge=0, le=100, description="Oil pressure in PSI")
    coolant_temp: float = Field(..., ge=0, le=120, description="Coolant temperature in Celsius")
    brake_temp: float = Field(..., ge=0, le=300, description="Brake temperature in Celsius")

class SensorDataBatch(BaseModel):
    """Batch of sensor data points."""
    data: List[SensorDataPoint] = Field(..., description="List of sensor data points")

class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request model."""
    data: List[SensorDataPoint] = Field(..., description="Sensor data for anomaly detection")
    detection_method: Optional[str] = Field("isolation_forest", description="Detection method")
    threshold: Optional[float] = Field(None, description="Custom threshold for detection")

class AnomalyResult(BaseModel):
    """Anomaly detection result for a single data point."""
    vehicle_id: str
    timestamp: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float

class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response model."""
    total_records: int
    anomalies_detected: int
    anomaly_rate: float
    results: List[AnomalyResult]
    processing_time: str

class PipelineStatus(BaseModel):
    """Pipeline status model."""
    status: str
    last_execution: Optional[str]
    total_records_processed: int
    total_anomalies_detected: int
    database_stats: Dict[str, Any]

class GenerateDataRequest(BaseModel):
    """Data generation request model."""
    num_records: int = Field(1000, ge=10, le=100000, description="Number of records to generate")
    num_vehicles: int = Field(5, ge=1, le=50, description="Number of vehicles")
    anomaly_rate: float = Field(0.05, ge=0, le=0.5, description="Anomaly rate (0-0.5)")

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AutoDataPipeline API",
        "version": "1.0.0",
        "description": "Vehicle sensor data anomaly detection service",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        stats = pipeline.storage.get_statistics()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "total_records": str(stats.get('total_records', 0))
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/status", response_model=PipelineStatus)
async def get_pipeline_status():
    """Get current pipeline status and statistics."""
    try:
        stats = pipeline.storage.get_statistics()
        
        return PipelineStatus(
            status="operational",
            last_execution=stats.get('last_update', 'Never'),
            total_records_processed=stats.get('total_records', 0),
            total_anomalies_detected=stats.get('total_anomalies', 0),
            database_stats=stats
        )
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-anomalies", response_model=AnomalyDetectionResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Detect anomalies in provided sensor data."""
    try:
        start_time = datetime.now()
        logger.info(f"Processing anomaly detection request for {len(request.data)} records")
        
        # Convert request data to DataFrame
        data_dict = [point.dict() for point in request.data]
        df = pd.DataFrame(data_dict)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Process data through pipeline
        processed_data = pipeline.process_data(df)
        
        # Detect anomalies
        anomalies = pipeline.detect_anomalies(processed_data)
        
        # Prepare results
        results = []
        # Determine anomaly column
        anomaly_col = None
        for col in ['anomaly_combined', 'anomaly_isolation_forest', 'anomaly_threshold', 'is_anomaly']:
            if col in anomalies.columns:
                anomaly_col = col
                break
        
        for _, row in anomalies.iterrows():
            is_anomaly_val = bool(row[anomaly_col]) if anomaly_col else False
            results.append(AnomalyResult(
                vehicle_id=row['vehicle_id'],
                timestamp=row['timestamp'].isoformat(),
                is_anomaly=is_anomaly_val,
                anomaly_score=float(row.get('isolation_forest_score', row.get('anomaly_score', 0))),
                confidence=float(row.get('confidence', 0))
            ))
        
        # Calculate statistics
        total_records = len(anomalies)
        if anomaly_col:
            anomalies_detected = len(anomalies[anomalies[anomaly_col] == 1])
        else:
            anomalies_detected = 0
        anomaly_rate = (anomalies_detected / total_records) * 100 if total_records > 0 else 0
        
        processing_time = str(datetime.now() - start_time)
        
        logger.info(f"Anomaly detection completed: {anomalies_detected}/{total_records} anomalies")
        
        return AnomalyDetectionResponse(
            total_records=total_records,
            anomalies_detected=anomalies_detected,
            anomaly_rate=anomaly_rate,
            results=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_data_file(file: UploadFile = File(...)):
    """Upload and process a data file."""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.csv', '.json')):
            raise HTTPException(status_code=400, detail="Only CSV and JSON files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process the file
            results = pipeline.run_processing_only(tmp_file_path)
            
            return {
                "message": "File processed successfully",
                "filename": file.filename,
                "results": results
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-data")
async def generate_sample_data(request: GenerateDataRequest):
    """Generate sample sensor data."""
    try:
        logger.info(f"Generating {request.num_records} sample records")
        
        # Generate data
        data = pipeline.ingestion.generate_synthetic_data(
            num_records=request.num_records,
            start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            anomaly_rate=request.anomaly_rate
        )
        
        # Save to temporary file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"generated_data_{timestamp}.csv"
        file_path = os.path.join("data", filename)
        
        os.makedirs("data", exist_ok=True)
        pipeline.ingestion.save_to_csv(data, file_path)
        
        return {
            "message": "Sample data generated successfully",
            "records_generated": len(data),
            "vehicles": data['vehicle_id'].nunique(),
            "file_path": file_path,
            "download_url": f"/download/{filename}"
        }
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated data files."""
    try:
        file_path = os.path.join("data", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-pipeline")
async def run_full_pipeline(background_tasks: BackgroundTasks, 
                           num_records: Optional[int] = 1000):
    """Run the complete data pipeline in background."""
    try:
        def run_pipeline_task():
            try:
                results = pipeline.run_full_pipeline(num_records=num_records)
                logger.info(f"Background pipeline execution completed: {results}")
            except Exception as e:
                logger.error(f"Background pipeline execution failed: {e}")
        
        background_tasks.add_task(run_pipeline_task)
        
        return {
            "message": "Pipeline execution started in background",
            "num_records": num_records,
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get database and processing statistics."""
    try:
        stats = pipeline.storage.get_statistics()
        
        # Add additional computed statistics
        if stats.get('total_records', 0) > 0 and stats.get('total_anomalies', 0) > 0:
            stats['overall_anomaly_rate'] = (stats['total_anomalies'] / stats['total_records']) * 100
        else:
            stats['overall_anomaly_rate'] = 0
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup")
async def cleanup_old_data(days_to_keep: int = 30):
    """Clean up old data from database."""
    try:
        if days_to_keep < 1:
            raise HTTPException(status_code=400, detail="days_to_keep must be at least 1")
        
        pipeline.cleanup_old_data(days_to_keep)
        
        return {
            "message": f"Data cleanup completed",
            "days_kept": days_to_keep
        }
        
    except Exception as e:
        logger.error(f"Error in data cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("AutoDataPipeline API starting up")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    logger.info("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("AutoDataPipeline API shutting down")

def run_api_server():
    """Run the FastAPI server."""
    logger.info(f"Starting API server on {API_CONFIG['host']}:{API_CONFIG['port']}")
    
    uvicorn.run(
        "src.api:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=API_CONFIG['debug'],
        log_level="info" if not API_CONFIG['debug'] else "debug"
    )

if __name__ == "__main__":
    run_api_server()