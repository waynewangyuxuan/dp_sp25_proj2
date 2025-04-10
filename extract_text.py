import re
import pandas as pd
import os
from pathlib import Path

# Path to pickle file
PICKLE_FILE = "/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.pkl"
CSV_OUTPUT = "/scratch/yw5954/dp_sp25_proj2/data/test/extracted_texts.csv"

def extract_text_from_pickle():
    """Extract text from pickle file without deserializing"""
    print(f"Reading binary content from {PICKLE_FILE}")
    with open(PICKLE_FILE, 'rb') as f:
        content = f.read()
    
    # Get file size
    file_size = len(content)
    print(f"File size: {file_size:,} bytes")
    
    # Check for some pickle protocol markers
    pickle_v2 = content.count(b'\x80\x02')
    pickle_v3 = content.count(b'\x80\x03')
    pickle_v4 = content.count(b'\x80\x04')
    pickle_v5 = content.count(b'\x80\x05')
    
    print(f"Pickle protocol markers found:")
    print(f"  Protocol 2: {pickle_v2}")
    print(f"  Protocol 3: {pickle_v3}")
    print(f"  Protocol 4: {pickle_v4}")
    print(f"  Protocol 5: {pickle_v5}")
    
    # Convert to text with lenient decoding
    text = content.decode('utf-8', errors='ignore')
    
    # Extract text that looks like news articles
    print("Extracting text that looks like news articles...")
    
    # Use a regex pattern to find items that look like text
    patterns = [
        # Standard sentences
        r'([A-Z][a-zA-Z0-9\s,\'\"\-\(\)]{20,350}[\.!?])',
        # Find quoted text (often contains articles)
        r'"([^"]{20,500})"',
        # Find text between single quotes
        r"'([^']{20,500})'"
    ]
    
    all_matches = []
    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, text)
        print(f"Pattern {i+1}: Found {len(matches)} matches")
        all_matches.extend(matches)
    
    # Remove duplicates (converting to a set and back to a list)
    all_matches = list(set(all_matches))
    print(f"After removing duplicates: {len(all_matches)} matches")
    
    # Filter out matches that don't look like news articles
    filtered_matches = []
    for match in all_matches:
        # Skip if it contains programming-related terms
        if any(term in match for term in ['function(', 'import ', 'class ', '.py', 'def ', '{', '}', '=>', '_.map']):
            continue
        
        # Skip if it has too many special characters
        special_chars = sum(1 for c in match if c in '@#$%^&*_+=`~|\\<>')
        if special_chars > len(match) * 0.05:  # More than 5% special chars
            continue
        
        # Skip if it doesn't have enough words
        if len(match.split()) < 5:
            continue
        
        # Add to filtered list
        filtered_matches.append(match.strip())
    
    print(f"After filtering: {len(filtered_matches)} potential articles")
    
    # Check for total expected AG News samples
    expected_samples = 7600  # Standard AG News test set size
    
    print(f"\nSAMPLE TEXTS (first 5):")
    for i in range(min(5, len(filtered_matches))):
        print(f"[{i+1}] {filtered_matches[i][:100]}...")
    
    # Save to CSV
    df = pd.DataFrame({
        'ID': range(len(filtered_matches)),
        'text': filtered_matches
    })
    
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"\nSaved {len(filtered_matches)} texts to {CSV_OUTPUT}")
    
    # Also save the exact number expected for AG News if needed
    if len(filtered_matches) > expected_samples:
        expected_csv = CSV_OUTPUT.replace('.csv', f'_{expected_samples}.csv')
        df_expected = pd.DataFrame({
            'ID': range(expected_samples),
            'text': filtered_matches[:expected_samples]
        })
        df_expected.to_csv(expected_csv, index=False)
        print(f"Also saved exactly {expected_samples} texts to {expected_csv}")
    
    return filtered_matches

if __name__ == "__main__":
    extract_text_from_pickle() 