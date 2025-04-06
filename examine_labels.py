import gzip
import pickle
import json

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

# Load and examine the 2000-class test file
test_data = load_dataset_file("data/WLASL/labels-100.test")
print("Data loaded ",test_data )
# Load vocab for reference
with open("data/WLASL/gloss_vocab.json", 'r') as f:
    vocab = json.load(f)

# Print structure info
print("Type of loaded data:", type(test_data))
print("\nNumber of entries:", len(test_data))

# Find and print entries with non-empty gloss
print("\nEntries with non-empty gloss:")
count = 0
for key, value in test_data.items():
    if value.get('gloss') and count < 5:  # Show first 5 entries with non-empty gloss
        print(f"\nEntry {count+1}:")
        print("Key:", key)
        print("Value type:", type(value))
        print("Value contents:", value)
        gloss = value['gloss']
        if isinstance(gloss, list):
            gloss = gloss[0]
        class_idx = vocab.get(gloss, "Not found")
        print(f"Gloss: {gloss} (Class index: {class_idx})")
        count += 1

# Print class distribution
class_counts = {}
for value in test_data.values():
    gloss = value.get('gloss')
    if gloss:
        if isinstance(gloss, list):
            gloss = gloss[0]
        class_idx = vocab.get(gloss, -1)
        if class_idx != -1:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

print("\nClass distribution:")
print(f"Number of unique classes in data: {len(class_counts)}")
print(f"Classes with indices 0-99: {len([i for i in class_counts.keys() if 0 <= i < 100])}")
print(f"Classes with indices 0-299: {len([i for i in class_counts.keys() if 0 <= i < 300])}")
print(f"Classes with indices 0-999: {len([i for i in class_counts.keys() if 0 <= i < 1000])}")