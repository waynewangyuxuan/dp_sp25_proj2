import pandas as pd
from datasets import Dataset
import os
import sys

# Define paths
PICKLE_FILE = "/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.pkl"
CSV_OUTPUT = "/scratch/yw5954/dp_sp25_proj2/data/test/test_data_8000.csv"

def convert_pickle_to_csv():
    """Convert the pickle file to CSV format"""
    print(f"Loading pickle file from: {PICKLE_FILE}")
    
    try:
        # Load the dataset
        dataset = pd.read_pickle(PICKLE_FILE)
        print(f"Successfully loaded dataset with type: {type(dataset)}")
        
        if isinstance(dataset, Dataset):
            print(f"Dataset has {len(dataset)} examples")
            print(f"Dataset features: {dataset.features}")
            print(f"Dataset column names: {dataset.column_names}")
            
            # Convert to dataframe
            df = dataset.to_pandas()
            print(f"Converted to pandas DataFrame with shape: {df.shape}")
        else:
            print("Unknown dataset type, attempting conversion...")
            df = pd.DataFrame(dataset)
        
        # Save to CSV
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"Successfully saved to CSV: {CSV_OUTPUT}")
        
        # Print sample data
        print("\nSample data (first 5 rows):")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"Error converting to CSV: {e}")
        return False

if __name__ == "__main__":
    # Convert pickle to CSV
    success = convert_pickle_to_csv()
    
    if success:
        print("\nNow you can downgrade NumPy with:")
        print("pip install numpy==1.24.4")
    else:
        print("\nConversion failed. Cannot proceed with downgrading NumPy.") 