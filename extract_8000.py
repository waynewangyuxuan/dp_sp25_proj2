import os
import sys
import re
import pandas as pd

# Paths
input_file = "/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.pkl"
output_csv = "/scratch/yw5954/dp_sp25_proj2/data/test/test_8000.csv"

def extract_8000_entries():
    """Extract exactly 8000 entries from the pickle file"""
    try:
        print(f"Reading binary content from {input_file}")
        with open(input_file, 'rb') as f:
            binary_content = f.read()
        
        # Decode as text with error handling
        text = binary_content.decode('utf-8', errors='ignore')
        
        # Pattern for news articles:
        # - Start with capital letter
        # - Contains alphanumeric chars, spaces, punctuation
        # - Ends with period, exclamation or question mark
        # - Reasonable length (20-350 chars)
        news_pattern = r'([A-Z][a-zA-Z0-9\s,\'\"\-\(\)]{20,350}[\.!?])'
        news_matches = re.findall(news_pattern, text)
        
        print(f"Found {len(news_matches)} potential entries")
        
        # Filter for higher quality entries
        filtered_entries = []
        
        for article in news_matches:
            # Skip entries that look like programming artifacts
            if any(x in article for x in ['function(', 'import ', 'class ', '.py', 'def ', '{', '}', '=>']):
                continue
            
            # Skip if it has too many special characters
            special_chars = sum(1 for c in article if c in '@#$%^&*_+=`~|\\<>')
            if special_chars > len(article) * 0.05:  # More than 5% special chars
                continue
                
            # Skip very short entries
            if len(article.split()) < 8:
                continue
                
            # Add to filtered list
            filtered_entries.append(article.strip())
        
        print(f"Filtered down to {len(filtered_entries)} quality entries")
        
        # Sort by length to prioritize more substantial entries
        filtered_entries.sort(key=len, reverse=True)
        
        # Get exactly 8000 entries (or as many as available if less than 8000)
        target_count = min(8000, len(filtered_entries))
        test_samples = filtered_entries[:target_count]
        
        print(f"Selected {len(test_samples)} entries for test set")
        
        # Save to CSV
        df = pd.DataFrame({
            'ID': range(len(test_samples)),
            'text': test_samples
        })
        df.to_csv(output_csv, index=False)
        
        print(f"Saved {len(test_samples)} entries to {output_csv}")
        print("\nSample entries:")
        for i in range(min(5, len(test_samples))):
            print(f"[{i}] {test_samples[i][:100]}...")
        
        return test_samples
    
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Extracting 8000 entries from pickle file...")
    extract_8000_entries() 