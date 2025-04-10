import pickle
import pandas as pd
import os
import sys
import json
import re

# Paths
input_file = "/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.pkl"
output_csv = "/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.csv"

# Function to read pickle file as text
def read_pickle_as_text():
    try:
        with open(input_file, 'rb') as f:
            # Read the raw bytes
            content = f.read()
            
            # Try to decode using various methods
            try:
                # Try to find text patterns in binary data
                text_content = content.decode('utf-8', errors='ignore')
                
                # Extract strings that look like sentences
                sentences = re.findall(r'[A-Za-z][^.!?]*[.!?]', text_content)
                
                if sentences:
                    print(f"Found {len(sentences)} potential text samples")
                    return sentences
                else:
                    print("No sentences found in text content")
                    return None
            except Exception as e:
                print(f"Error parsing text: {e}")
                return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Extract the data
print(f"Attempting to extract text data from {input_file}")
extracted_data = read_pickle_as_text()

if extracted_data and len(extracted_data) > 0:
    # Create a DataFrame with ID and text
    df = pd.DataFrame({
        'ID': range(len(extracted_data)),
        'text': extracted_data
    })
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Successfully converted to CSV: {output_csv}")
    print(f"Sample data (first 5 rows):")
    print(df.head())
else:
    print("Failed to extract text data from pickle file")
    sys.exit(1) 