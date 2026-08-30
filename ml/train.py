import os
import json
import urllib.request
import zipfile

# Fix invalid JAVA_HOME environment variable that causes PySpark to hang
if os.environ.get('JAVA_HOME', '').endswith('bin'):
    os.environ['JAVA_HOME'] = os.environ['JAVA_HOME'][:-4]
elif not os.environ.get('JAVA_HOME'):
    os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'
    
# Fix PySpark missing winutils.exe on Windows which causes Python worker SocketException
os.environ['HADOOP_HOME'] = r'e:\CSE761\hadoop'
os.environ['PATH'] = r'e:\CSE761\hadoop\bin' + os.pathsep + os.environ.get('PATH', '')

import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, hour, dayofweek, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def download_and_extract_shapefile(data_dir):
    """Downloads the official taxi zone shapefile and calculates centroids using geopandas."""
    shp_zip = os.path.join(data_dir, "taxi_zones.zip")
    if not os.path.exists(shp_zip):
        print("Downloading taxi zone shapefile...")
        url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
        urllib.request.urlretrieve(url, shp_zip)
        with zipfile.ZipFile(shp_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_dir, "taxi_zones"))
            
    try:
        import geopandas as gpd
        shp_path = os.path.join(data_dir, "taxi_zones", "taxi_zones", "taxi_zones.shp")
        gdf = gpd.read_file(shp_path)
        gdf = gdf.to_crs(epsg=4326) # Convert to standard Lat/Lon (WGS84)
        gdf['centroid'] = gdf.geometry.centroid
        gdf['longitude'] = gdf.centroid.x
        gdf['latitude'] = gdf.centroid.y
        # Save mapping to CSV
        csv_path = os.path.join(data_dir, "taxi_zones_centroids.csv")
        gdf[['LocationID', 'zone', 'borough', 'longitude', 'latitude']].to_csv(csv_path, index=False)
        print(f"Centroids saved to {csv_path}")
        return csv_path
    except ImportError:
        print("Geopandas not installed. Please install it to process shapefiles.")
        return None

def train_model():
    print("Initializing PySpark Session for Urban Mobility Predictor...")
    spark = SparkSession.builder \
        .appName("UrbanMobilityPredictor") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    backend_dir = os.path.join(base_dir, "backend")
    metrics_dir = os.path.join(base_dir, "frontend", "public", "metrics")
    
    os.makedirs(backend_dir, exist_ok=True)
    os.makedirs(os.path.join(backend_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(backend_dir, "data"), exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 1. Download shapefile and get centroids
    centroids_csv = download_and_extract_shapefile(data_dir)
    
    # 2. Load the downloaded parquet file using Pandas to bypass the Windows Hadoop NativeIO bug
    parquet_file = os.path.join(data_dir, "yellow_tripdata_2023-01.parquet")
    print(f"Loading data from {parquet_file}...")
    try:
        pdf = pd.read_parquet(parquet_file, engine='fastparquet')
        print("Data loaded into Pandas. Sampling 50,000 rows for local demo...")
        pdf = pdf.head(50000)
        df = spark.createDataFrame(pdf)
    except Exception as e:
        print(f"Error reading parquet files: {e}")
        return
        
    print(f"Total raw records: {df.count()}")
    
    from pyspark.sql.functions import unix_timestamp, hour, dayofweek, col
    # 3. Data Cleaning and Feature Engineering
    df = df.withColumn("trip_duration_mins", 
                      (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60.0)
                      
    df = df.filter((col("trip_duration_mins") > 2) & (col("trip_duration_mins") < 120))
    df = df.filter((col("trip_distance") > 0) & (col("trip_distance") < 100))
    
    df = df.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
    df = df.withColumn("pickup_dow", dayofweek("tpep_pickup_datetime"))
    
    required_cols = ["trip_distance", "PULocationID", "DOLocationID", "pickup_hour", "pickup_dow", "trip_duration_mins"]
    df = df.select(required_cols).dropna()
    
    print("Data cleaning complete. Training model...")
    
    # 4. Machine Learning Pipeline & Train/Test Split
    feature_cols = ["trip_distance", "PULocationID", "DOLocationID", "pickup_hour", "pickup_dow"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    rf = RandomForestRegressor(featuresCol="features", labelCol="trip_duration_mins", numTrees=20, maxDepth=10)
    pipeline = Pipeline(stages=[assembler, rf])
    
    # Split for IEEE Paper evaluation
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Train model
    model = pipeline.fit(train_df)
    
    # Save the model
    model_path = os.path.join(backend_dir, "model", "rf_model")
    model.write().overwrite().save(model_path)
    
    # 5. Evaluate and Generate Metrics for IEEE Paper
    print("Evaluating Model for Metrics...")
    predictions = model.transform(test_df)
    
    eval_rmse = RegressionEvaluator(labelCol="trip_duration_mins", predictionCol="prediction", metricName="rmse")
    eval_r2 = RegressionEvaluator(labelCol="trip_duration_mins", predictionCol="prediction", metricName="r2")
    eval_mae = RegressionEvaluator(labelCol="trip_duration_mins", predictionCol="prediction", metricName="mae")
    
    rmse = eval_rmse.evaluate(predictions)
    r2 = eval_r2.evaluate(predictions)
    mae = eval_mae.evaluate(predictions)
    
    metrics = {
        "RMSE": round(rmse, 2),
        "R2_Score": round(r2, 4),
        "MAE": round(mae, 2)
    }
    
    with open(os.path.join(backend_dir, "data", "metrics.json"), "w") as f:
        json.dump(metrics, f)
        
    print(f"Metrics saved: {metrics}")
    
    # 5.5 Extract 20 random test samples for frontend showcase
    print("Extracting test samples for frontend showcase...")
    sample_df = test_df.select("trip_distance", "PULocationID", "DOLocationID", "pickup_hour", "pickup_dow", "trip_duration_mins").sample(withReplacement=False, fraction=0.05, seed=42).limit(20)
    sample_pd = sample_df.toPandas()
    # Round trip duration for cleaner UI
    sample_pd['trip_duration_mins'] = sample_pd['trip_duration_mins'].round(2)
    sample_json = sample_pd.to_dict(orient='records')
    with open(os.path.join(backend_dir, "data", "test_samples.json"), "w") as f:
        json.dump(sample_json, f)
    
    # 6. Generate Graphs (Feature Importance & Actual vs Predicted)
    print("Generating Graphs...")
    # Extract Feature Importances
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_cols, palette="viridis")
    plt.title("Feature Importances in RandomForest Model")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "feature_importance.png"))
    plt.close()
    
    # Actual vs Predicted Scatter Plot (sample 2000 points so it doesn't crash memory)
    scatter_data = predictions.select("trip_duration_mins", "prediction").limit(2000).toPandas()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="trip_duration_mins", y="prediction", data=scatter_data, alpha=0.5, color="blue")
    plt.plot([0, 120], [0, 120], 'r--') # Perfect prediction line
    plt.title("Actual vs. Predicted Trip Duration")
    plt.xlabel("Actual Duration (mins)")
    plt.ylabel("Predicted Duration (mins)")
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "actual_vs_predicted.png"))
    plt.close()

    # 7. Generate Hotspots Data
    print("Generating Hotspots aggregation for frontend visualization...")
    hotspots = df.groupBy("PULocationID").agg(count("*").alias("trip_count")).orderBy(col("trip_count").desc()).limit(100)
    hotspots_pd = hotspots.toPandas()
    
    if centroids_csv and os.path.exists(centroids_csv):
        centroids_pd = pd.read_csv(centroids_csv)
        hotspots_pd = pd.merge(hotspots_pd, centroids_pd, left_on='PULocationID', right_on='LocationID', how='inner')
        hotspots_json = hotspots_pd[['LocationID', 'zone', 'latitude', 'longitude', 'trip_count']].to_dict(orient='records')
    else:
        hotspots_json = hotspots_pd.to_dict(orient='records')
        
    hotspots_path = os.path.join(backend_dir, "data", "hotspots.json")
    with open(hotspots_path, "w") as f:
        json.dump(hotspots_json, f)
        
    spark.stop()
    print("Pipeline fully completed!")

if __name__ == "__main__":
    train_model()
