import sys
import pickle
import pandas as pd
import numpy as np

PICKLE_FILE = "/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.pkl"

# Try different approaches to load the pickle file
methods = [
    ("pandas.read_pickle", lambda: pd.read_pickle(PICKLE_FILE)),
    ("pickle.load", lambda: pickle.load(open(PICKLE_FILE, "rb"))),
    ("pickle.load with latin1", lambda: pickle.load(open(PICKLE_FILE, "rb"), encoding="latin1")),
    ("pickle.load with bytes", lambda: pickle.load(open(PICKLE_FILE, "rb"), encoding="bytes")),
    ("numpy.load", lambda: np.load(PICKLE_FILE, allow_pickle=True))
]

def examine_data(data, method_name):
    """Examine the structure of the loaded data"""
    print("\n" + "="*50)
    print(f"EXAMINING DATA LOADED WITH: {method_name}")
    print(f"Type: {type(data)}")
    
    # Handle different types
    if hasattr(data, "__len__"):
        print(f"Length: {len(data)}")
        
        # For large collections, check the first few items
        if len(data) > 0:
            if isinstance(data, dict):
                print("Keys:", list(data.keys())[:10])
                for key in list(data.keys())[:3]:
                    print(f"  Value for '{key}':", data[key])
                    if hasattr(data[key], "__len__") and len(data[key]) > 0:
                        print(f"    Type of first item in '{key}':", type(data[key][0]))
                        print(f"    First item in '{key}':", data[key][0])
            elif isinstance(data, (list, tuple)):
                print("First item type:", type(data[0]))
                print("First 3 items:")
                for i in range(min(3, len(data))):
                    item = data[i]
                    if isinstance(item, str) and len(item) > 100:
                        item = item[:100] + "..."
                    print(f"  [{i}]: {item}")
    else:
        print("Object doesn't have a length attribute")
    
    print("="*50)

# Try each method
for name, method in methods:
    try:
        print(f"Trying to load with {name}...")
        data = method()
        examine_data(data, name)
        
        # If successful, break
        print(f"SUCCESS: Successfully loaded with {name}")
        
        # Ask for confirmation to continue trying other methods
        response = input("Continue trying other methods? (y/n): ")
        if response.lower() != 'y':
            break
            
    except Exception as e:
        print(f"FAILED: {name} - Error: {e}")

print("\nDone examining pickle file") 