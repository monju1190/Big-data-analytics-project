import urllib.request
import os
import time

def download_data(year=2023):
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}-{:02d}.parquet"
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Downloading 1 year of Urban Mobility data for {year}...")
    
    for month in range(1, 13):
        url = base_url.format(year, month)
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"{filename} already exists. Skipping.")
            continue
            
        print(f"Downloading {filename}...")
        try:
            # Adding headers to simulate browser
            req = urllib.request.Request(
                url, 
                data=None, 
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            
        time.sleep(1) # Be nice to the server

if __name__ == "__main__":
    download_data(2023)
