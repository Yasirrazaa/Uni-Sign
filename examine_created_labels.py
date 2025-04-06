import gzip
import pickle
import json

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def examine_label_file(filename):
    print(f"\nExamining {filename}:")
    try:
        data = load_dataset_file(filename)
        print(f"Number of entries: {len(data)}")
        
        # Get a sample entry
        sample_key = list(data.keys())[0]
        print(f"Sample entry: {sample_key} -> {data[sample_key]}")
        
        # Count unique glosses
        glosses = [v.get('gloss', '') for v in data.values()]
        unique_glosses = set(glosses)
        print(f"Number of unique glosses: {len(unique_glosses)}")
        print(f"Sample glosses: {list(unique_glosses)[:10]}")
        
        # Count entries per gloss
        gloss_counts = {}
        for gloss in glosses:
            if gloss:
                gloss_counts[gloss] = gloss_counts.get(gloss, 0) + 1
        
        print(f"Top 5 glosses by count:")
        for gloss, count in sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {gloss}: {count} entries")
            
    except Exception as e:
        print(f"Error examining {filename}: {e}")

# Examine the created label files
for class_size in [100, 300, 1000]:
    for subset in ['train', 'val', 'test']:
        examine_label_file(f"data/WLASL/labels-{class_size}.{subset}")
