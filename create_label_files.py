import json
import gzip
import pickle
import os

def load_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_dataset_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with gzip.open(filename, "wb") as f:
        pickle.dump(data, f)

def create_label_files(json_file, class_limit, output_prefix):
    # Load the JSON file
    data = load_json_file(json_file)

    # Load the class list
    class_list = {}
    with open('wlasl_class_list.txt', 'r') as f:
        for line in f:
            idx, gloss = line.strip().split('\t')
            class_list[int(idx)] = gloss

    # Create dictionaries to store the label data for each subset
    train_data = {}
    val_data = {}
    test_data = {}

    # Process each video entry
    for video_id, video_info in data.items():
        subset = video_info.get('subset')
        action = video_info.get('action', [])

        # Check if the class is within our limit
        if len(action) >= 1 and action[0] < class_limit:
            # Create the label entry
            entry = {
                'name': video_id,
                'gloss': class_list.get(action[0], ''),  # Get the gloss from class_list
                'text': class_list.get(action[0], ''),   # Use the same for text
                'video_path': f'{video_id}.mp4'
            }

            # Add to the appropriate subset
            if subset == 'train':
                train_data[video_id] = entry
            elif subset == 'val':
                val_data[video_id] = entry
            elif subset == 'test':
                test_data[video_id] = entry

    # Save the label data for each subset
    save_dataset_file(train_data, f"{output_prefix}.train")
    save_dataset_file(val_data, f"{output_prefix}.val")
    save_dataset_file(test_data, f"{output_prefix}.test")

    print(f"Created label files for {class_limit} classes:")
    print(f"  {output_prefix}.train with {len(train_data)} entries")
    print(f"  {output_prefix}.val with {len(val_data)} entries")
    print(f"  {output_prefix}.test with {len(test_data)} entries")

def main():
    # Create label files for different subsets
    create_label_files('nslt_100.json', 100, 'data/WLASL/labels-100')
    create_label_files('nslt_300.json', 300, 'data/WLASL/labels-300')
    create_label_files('nslt_1000.json', 1000, 'data/WLASL/labels-1000')

if __name__ == "__main__":
    main()
