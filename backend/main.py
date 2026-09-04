import os
import sys

# Fix invalid JAVA_HOME environment variable that causes PySpark to hang
if os.environ.get('JAVA_HOME', '').endswith('bin'):
    os.environ['JAVA_HOME'] = os.environ['JAVA_HOME'][:-4]
elif not os.environ.get('JAVA_HOME'):
    os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'

# Fix PySpark missing winutils.exe on Windows
os.environ['HADOOP_HOME'] = r'e:\CSE761\hadoop'
os.environ['PATH'] = r'e:\CSE761\hadoop\bin' + os.pathsep + os.environ.get('PATH', '')

# Fix SocketException for Python Workers
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

app = FastAPI(title="Urban Mobility Predictor API")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for Spark and Model
spark = None
model = None
base_dir = os.path.dirname(os.path.abspath(__file__))
hotspots_path = os.path.join(base_dir, "data", "hotspots.json")
model_path = os.path.join(base_dir, "model", "rf_model")

@app.on_event("startup")
async def startup_event():
    global spark, model
    print("Initializing PySpark in FastAPI...")
    try:
        spark = SparkSession.builder \
            .appName("UrbanMobilityAPI") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
            
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = PipelineModel.load(model_path)
            print("Model loaded successfully.")
        else:
            print("Warning: Model not found. Run ml/train.py first.")
    except Exception as e:
        print(f"Error initializing Spark/Model: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global spark
    if spark:
        spark.stop()

class PredictionRequest(BaseModel):
    trip_distance: float
    PULocationID: int
    DOLocationID: int
    pickup_hour: int
    pickup_dow: int

@app.get("/api/hotspots")
async def get_hotspots():
    """Returns the pre-calculated hotspot aggregations from PySpark."""
    if not os.path.exists(hotspots_path):
        raise HTTPException(status_code=404, detail="Hotspots data not found. Run ml/train.py first.")
    
    with open(hotspots_path, "r") as f:
        data = json.load(f)
    return data

@app.get("/api/test-samples")
async def get_test_samples():
    """Returns real test trip samples for frontend showcase."""
    samples_path = os.path.join(base_dir, "data", "test_samples.json")
    if not os.path.exists(samples_path):
        raise HTTPException(status_code=404, detail="Test samples not found. Run ml/train.py first.")
    
    with open(samples_path, "r") as f:
        data = json.load(f)
    return data

@app.get("/api/metrics")
async def get_metrics():
    """Returns the evaluation metrics from the trained model."""
    metrics_path = os.path.join(base_dir, "data", "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found. Run ml/train.py first.")
    
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data

@app.get("/api/zones")
async def get_zones():
    """Returns mapping of LocationID to coordinates to help frontend compute distance."""
    csv_path = os.path.join(os.path.dirname(base_dir), "data", "taxi_zones_centroids.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Zones data not found at {csv_path}")
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    zones_dict = {}
    for _, row in df.iterrows():
        zones_dict[str(int(row['LocationID']))] = {
            "lat": float(row['latitude']),
            "lon": float(row['longitude'])
        }
    return zones_dict

@app.post("/api/predict")
async def predict_duration(req: PredictionRequest):
    """Runs a live prediction through the PySpark ML model."""
    if not model or not spark:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Create a DataFrame for the single request
        schema = StructType([
            StructField("trip_distance", DoubleType(), True),
            StructField("PULocationID", IntegerType(), True),
            StructField("DOLocationID", IntegerType(), True),
            StructField("pickup_hour", IntegerType(), True),
            StructField("pickup_dow", IntegerType(), True)
        ])
        
        data = [(req.trip_distance, req.PULocationID, req.DOLocationID, req.pickup_hour, req.pickup_dow)]
        df = spark.createDataFrame(data, schema)
        
        # Run prediction
        predictions = model.transform(df)
        predicted_mins = predictions.select("prediction").collect()[0][0]
        
        return {
            "predicted_duration_mins": round(predicted_mins, 2),
            "inputs": req.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
