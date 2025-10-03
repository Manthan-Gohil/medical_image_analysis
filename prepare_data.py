import pandas as pd
import os

def create_dataset_csv(root_dir, output_csv):
    """Creates a CSV file with image paths and labels from the MURA dataset structure."""
    print("Searching for dataset paths...")
    try:
        train_paths_csv = os.path.join(root_dir, 'MURA-v1.1/train_image_paths.csv')
        valid_paths_csv = os.path.join(root_dir, 'MURA-v1.1/valid_image_paths.csv')

        df_train = pd.read_csv(train_paths_csv, header=None, names=['image_path'])
        df_valid = pd.read_csv(valid_paths_csv, header=None, names=['image_path'])
    except FileNotFoundError:
        print("\n[ERROR] Ensure the MURA dataset is unzipped correctly inside the 'data' folder.")
        print(f"Expected path not found: {train_paths_csv}")
        return

    # Combine training and validation sets for this example
    df_all = pd.concat([df_train, df_valid], ignore_index=True)

    # Extract label from the path (1 for abnormal/fracture, 0 for normal)
    df_all['label'] = df_all['image_path'].apply(lambda x: 1 if 'positive' in x else 0)
    
    # Make image paths absolute from the project root
    df_all['image_path'] = df_all['image_path'].apply(lambda x: os.path.join(root_dir, x))

    df_all.to_csv(output_csv, index=False)
    print(f"\nSuccessfully created dataset CSV at '{output_csv}'")
    print(f"Total images found: {len(df_all)}")
    print("Label distribution:")
    print(df_all['label'].value_counts())

if __name__ == '__main__':
    # The 'MURA-v1.1' folder should be inside the 'data' directory
    dataset_root = 'data/'
    output_file = 'data/mura_dataset.csv'
    create_dataset_csv(dataset_root, output_file)