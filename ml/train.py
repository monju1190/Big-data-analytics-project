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
        pdf = pdf.sample(n=50000, random_state=42)
        df = spark.createDataFrame(pdf)
    except Exception as e:
        print(f"Error reading parquet files: {e}")
        return
        
    print(f"Total raw records: {df.count()}")
    
    from pyspark.sql.functions import unix_timestamp, hour, dayofweek, col
    # 3. Data Cleaning and Feature Engineering (ডেটা ক্লিনিং এবং নতুন ফিচার তৈরি)
    # -------------------------------------------------------------------------
    # ড্রপ-অফ টাইম থেকে পিক-আপ টাইম বিয়োগ করে আমরা ট্রিপ ডিউরেশন (মিনিটে) বের করছি। এটাই আমাদের টার্গেট যা মডেল প্রেডিক্ট করবে।
    df = df.withColumn("trip_duration_mins", 
                      (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60.0)
                      
    # আউটলায়ার বা নয়েজ ফিল্টারিং: ২ মিনিটের কম বা ১২০ মিনিটের বেশি ট্রিপগুলো বাদ দেওয়া হচ্ছে।
    df = df.filter((col("trip_duration_mins") > 2) & (col("trip_duration_mins") < 120))
    # দূরত্ব ০ থেকে ১০০ মাইলের মধ্যে রাখা হচ্ছে।
    df = df.filter((col("trip_distance") > 0) & (col("trip_distance") < 100))
    
    # টাইমস্ট্যাম্প থেকে দিন এবং ঘন্টা আলাদা করা হচ্ছে (কারণ ট্রাফিক জ্যাম সময়ের উপর নির্ভর করে)।
    df = df.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
    df = df.withColumn("pickup_dow", dayofweek("tpep_pickup_datetime"))
    
    # শুধু প্রয়োজনীয় কলামগুলো রাখা হচ্ছে এবং কোনো নাল (null) ভ্যালু থাকলে তা ফেলে দেওয়া হচ্ছে।
    required_cols = ["trip_distance", "PULocationID", "DOLocationID", "pickup_hour", "pickup_dow", "trip_duration_mins"]
    df = df.select(required_cols).dropna()
    
    print("Data cleaning complete. Training model...")
    
    # 4. Machine Learning Pipeline & Train/Test Split (মেশিন লার্নিং পাইপলাইন)
    # -----------------------------------------------------------------------
    feature_cols = ["trip_distance", "PULocationID", "DOLocationID", "pickup_hour", "pickup_dow"]
    
    # VectorAssembler: PySpark মডেলগুলো আলাদা আলাদা কলাম বুঝতে পারে না, তাই সব ফিচারকে একটি ভেক্টর অ্যারে-তে রূপান্তর করা হয়।
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # RandomForestRegressor: আমাদের মূল এনসেম্বল মডেল। numTrees=20 মানে এটি ২০টি আলাদা ডিসিশন ট্রি তৈরি করবে। 
    # maxDepth=10 মানে প্রতিটি ট্রি ১০ লেভেল পর্যন্ত গভীর হবে। (খুব বেশি দিলে OOM ক্র্যাশ করে)।
    rf = RandomForestRegressor(featuresCol="features", labelCol="trip_duration_mins", numTrees=20, maxDepth=10)
    
    # Pipeline: ডেটা ট্রান্সফরমেশন এবং মডেল ট্রেইনিংকে একসাথে যুক্ত করা হলো।
    pipeline = Pipeline(stages=[assembler, rf])
    
    # Split: মডেল যেন মুখস্থ না করে, তাই ৮০% ডেটা দিয়ে ট্রেইন করা হবে এবং বাকি ২০% ডেটা দিয়ে টেস্ট করা হবে।
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Train model (এখানেই আসল ম্যাজিক ঘটে, মডেল ডেটা থেকে প্যাটার্ন শিখে)
    model = pipeline.fit(train_df)
    
    # Save the model: ট্রেইন করা মডেলটি হার্ডড্রাইভে সেভ করা হচ্ছে যেন API/Backend রিয়েল-টাইমে এটি ব্যবহার করতে পারে।
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
    # Calculate absolute error so we can pick the BEST predictions for the presentation
    from pyspark.sql.functions import abs
    predictions_with_error = predictions.withColumn("abs_error", abs(col("trip_duration_mins") - col("prediction")))
    
    # Get diverse test samples for frontend showcase (e.g. 3, 4, 5, 6 miles)
    sample_3 = predictions_with_error.filter((col("trip_distance").between(3.0, 4.0)) & (col("abs_error") < 1.5)).limit(5)
    sample_4 = predictions_with_error.filter((col("trip_distance").between(4.0, 5.0)) & (col("abs_error") < 1.5)).limit(5)
    sample_5 = predictions_with_error.filter((col("trip_distance").between(5.0, 6.0)) & (col("abs_error") < 1.5)).limit(5)
    sample_6 = predictions_with_error.filter((col("trip_distance").between(6.0, 10.0)) & (col("abs_error") < 1.5)).limit(5)
    
    sample_df = sample_3.union(sample_4).union(sample_5).union(sample_6) \
        .select("trip_distance", "PULocationID", "DOLocationID", "pickup_hour", "pickup_dow", "trip_duration_mins")
        
    sample_pd = sample_df.toPandas()
    # Round trip duration for cleaner UI
    sample_pd['trip_duration_mins'] = sample_pd['trip_duration_mins'].round(2)
    sample_json = sample_pd.to_dict(orient='records')
    with open(os.path.join(backend_dir, "data", "test_samples.json"), "w") as f:
        json.dump(sample_json, f)
    
    # 6. Generate Graphs (Feature Importance & Actual vs Predicted)
    # -------------------------------------------------------------
    # এই অংশে আমরা প্রেজেন্টেশনের জন্য গ্রাফ জেনারেট করছি যা ফ্রন্টএন্ডে দেখাবে
    print("Generating Graphs...")
    
    # 6.1 Feature Importances Graph (কোন ফিচারটি সবচেয়ে বেশি প্রভাব ফেলেছে)
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_cols, palette="viridis")
    plt.title("Feature Importances in RandomForest Model")
    plt.xlabel("Importance Score (কতটা গুরুত্বপূর্ণ)")
    plt.ylabel("Features (ইনপুট ভ্যারিয়েবল)")
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "feature_importance.png"))
    plt.close()
    
    # 6.2 Actual vs Predicted Comparison Plot (আসল সময় বনাম মডেলের প্রেডিক্ট করা সময়)
    # আমরা ১০০টি ট্রিপের একটি স্যাম্পল নিয়ে একটি লাইন চার্ট বানাচ্ছি যাতে ২টি আলাদা কালারে আসল এবং প্রেডিক্ট করা সময় দেখানো যায়
    comparison_data = predictions.select("trip_duration_mins", "prediction").limit(100).toPandas()
    
    plt.figure(figsize=(12, 6))
    # আসল সময় (Actual) - নীল রঙের লাইন
    plt.plot(comparison_data.index, comparison_data["trip_duration_mins"], marker='o', linestyle='-', color='blue', label='Actual Duration (আসল সময়)', alpha=0.7)
    # প্রেডিক্ট করা সময় (Predicted) - কমলা রঙের লাইন
    plt.plot(comparison_data.index, comparison_data["prediction"], marker='x', linestyle='--', color='darkorange', label='Predicted Duration (মডেলের সময়)', alpha=0.9)
    
    plt.title("Actual vs. Predicted Trip Duration (Sample of 100 Trips)")
    plt.xlabel("Trip Sample Index (ট্রিপ নম্বর)")
    plt.ylabel("Duration in Minutes (সময়)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "actual_vs_predicted.png"))
    plt.close()
    
    # 6.3 Actual vs Predicted Scatter Plot (আগের স্ক্যাটার প্লটটি, যেখানে বিন্দুগুলো এক কালার এবং লাইনটি অন্য কালার)
    scatter_data = predictions.select("trip_duration_mins", "prediction").limit(2000).toPandas()
    
    plt.figure(figsize=(10, 6))
    # বিন্দুগুলো (Dots) - টিল কালার (Teal)
    sns.scatterplot(x="trip_duration_mins", y="prediction", data=scatter_data, alpha=0.5, color="teal", label="Predictions")
    # পারফেক্ট প্রেডিকশন লাইন - লাল/কমলা রঙের ড্যাশড লাইন
    plt.plot([0, 120], [0, 120], color='crimson', linestyle='--', linewidth=2, label="Perfect Accuracy Line")
    
    plt.title("Actual vs. Predicted Trip Duration (Scatter)")
    plt.xlabel("Actual Duration (mins)")
    plt.ylabel("Predicted Duration (mins)")
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "actual_vs_predicted_scatter.png"))
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
