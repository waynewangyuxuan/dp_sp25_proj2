import pickle
import sys

try:
    # Try with latin1 encoding first
    with open("/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")
        print("Loaded with latin1 encoding")
except Exception as e:
    print(f"Error with latin1 encoding: {e}")
    try:
        # Try with bytes encoding
        with open("/scratch/yw5954/dp_sp25_proj2/data/test/test_unlabelled.pkl", "rb") as f:
            data = pickle.load(f)
            print("Loaded with default encoding")
    except Exception as e:
        print(f"Error with default encoding: {e}")
        sys.exit(1)

print(f"Data type: {type(data)}")
print(f"Data length: {len(data) if hasattr(data, '__len__') else 'No length'}")

if isinstance(data, (list, tuple)) and len(data) > 0:
    print(f"First item type: {type(data[0])}")
    print(f"First item: {data[0]}")
elif isinstance(data, dict):
    print(f"Keys: {data.keys()}")
    for k in data:
        print(f"Key: {k}, Type: {type(data[k])}")
        if hasattr(data[k], '__len__'):
            print(f"Length: {len(data[k])}")
            if len(data[k]) > 0:
                print(f"First item type: {type(data[k][0])}")
                print(f"First item: {data[k][0]}")
else:
    print("Cannot inspect data further")
