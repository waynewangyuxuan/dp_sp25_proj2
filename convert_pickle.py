import os
import sys
import io
import pandas as pd
import re
import json

# Paths
input_file = "/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.pkl"
output_jsonl = "/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.jsonl"

def extract_news_articles():
    """Extract news-like articles from the pickle file"""
    try:
        # Read the binary file
        with open(input_file, 'rb') as f:
            raw_content = f.read()
        
        # Convert to text with lenient decoding
        text_content = raw_content.decode('utf-8', errors='ignore')
        
        # Look for news article patterns - AG News has a distinctive format
        # News articles are often 1-3 sentences, with the first being the headline
        # Use regex to find coherent looking paragraphs
        
        # This regex looks for potential news articles:
        # - Start with capital letter
        # - Most news has quotes, numbers, proper nouns
        # - Should end with punctuation
        # - Length should be reasonable
        potential_articles = re.findall(r'([A-Z][a-zA-Z0-9\s,\'\"\-\(\)]{20,200}[\.!?])', text_content)
        
        # Filter out low-quality matches
        filtered_articles = []
        for article in potential_articles:
            # Skip entries that look like programming artifacts
            if any(x in article for x in ['function(', 'import ', 'class ', '.py', 'def ', '{', '}', '=>']):
                continue
            
            # Skip if it has too many special characters
            special_chars = sum(1 for c in article if c in '@#$%^&*_+=`~|\\<>')
            if special_chars > len(article) * 0.05:  # More than 5% special chars
                continue
                
            # Add to filtered list
            filtered_articles.append(article.strip())
        
        print(f"Found {len(filtered_articles)} potential news articles")
        
        # Take the first 7600 items (AG News test set has 7600 examples)
        # Or take as many as available if less than 7600
        test_samples = filtered_articles[:min(7600, len(filtered_articles))]
        
        # Save to JSONL format (more flexible than CSV for text)
        with open(output_jsonl, 'w') as f:
            for i, text in enumerate(test_samples):
                json_obj = {"id": i, "text": text}
                f.write(json.dumps(json_obj) + '\n')
        
        print(f"Saved {len(test_samples)} text samples to {output_jsonl}")
        
        # Show some examples
        print("\nSample entries:")
        for i in range(min(5, len(test_samples))):
            print(f"[{i}] {test_samples[i][:100]}...")
            
        return test_samples
    
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    extracted_texts = extract_news_articles()
    
    if not extracted_texts:
        print("Failed to extract texts from pickle file")
        sys.exit(1)
    
    # Also create a compatible CSV for prediction
    csv_path = input_file.replace('.pkl', '.csv')
    df = pd.DataFrame({
        'ID': range(len(extracted_texts)),
        'text': extracted_texts
    })
    df.to_csv(csv_path, index=False)
    print(f"Also saved data to CSV: {csv_path}")
    
    print("\nNext step: Use the CSV file with the predict.py script") 