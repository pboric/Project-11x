import pandas as pd
import numpy as np
import requests
import json
from time import time

API_URL = "https://home-credit-group-defaulter-prediction.onrender.com"

def clean_for_json(obj):
    """Clean data to make it JSON serializable"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return 0
        return float(obj)
    else:
        return obj

def test_with_data():
    print("Loading test data...")
    test_data = pd.read_csv(r'C:\Users\kresi\OneDrive\Desktop\Turing college\Project11xx\processed_test.csv')
    
    test_data = test_data.replace([np.inf, -np.inf], np.nan)
    test_data = test_data.fillna(0)
    
    print(f"Loaded {test_data.shape[0]} rows and {test_data.shape[1]} features")
    
    print("\nTesting single prediction...")
    sample = test_data.iloc[0].to_dict()
    sample = clean_for_json(sample)
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"data": sample}
        )
        
        if response.status_code == 200:
            print("\nSingle Prediction Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
        
        print("\nTesting batch prediction...")
        batch_samples = test_data.head(5).to_dict('records')
        batch_samples = clean_for_json(batch_samples)
        
        batch_response = requests.post(
            f"{API_URL}/batch-predict",
            json={"data": batch_samples}
        )
        
        if batch_response.status_code == 200:
            print("\nBatch Prediction Response:")
            print(json.dumps(batch_response.json(), indent=2))
        else:
            print(f"Error: {batch_response.status_code}")
            print(batch_response.text)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
def print_data_sample(data):
    """Print a sample of the data for debugging"""
    sample = data.iloc[0]
    print("\nFirst row data sample:")
    non_zero = sample[sample != 0]
    print(non_zero.head())
    print(f"\nNumber of non-zero features: {len(non_zero)}")

if __name__ == "__main__":
    print("Testing Home Credit Default Prediction API")
    print("=" * 50)
    
    print("\nTesting API connection...")
    try:
        response = requests.get(API_URL)
        print("API Status:", response.json())
    except Exception as e:
        print(f"Connection Error: {str(e)}")
    
    test_with_data()