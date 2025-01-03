import os
import shutil
import zipfile

# Paths
RAW_DATA_DIR = "data/raw/PKLot.v2-640.yolov8-obb.zip"
PREPROCESSED_DATA_DIR = "data/preprocessed"
DATA_YAML_PATH = os.path.join(PREPROCESSED_DATA_DIR, "data.yaml")

def unzip_dataset(zip_path, output_dir):
    """
    Unzips the dataset into the specified directory.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip file not found: {zip_path}")
    
    if os.path.exists(output_dir):
        print(f"Removing existing preprocessed directory: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"Unzipping dataset from {zip_path} to {output_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Dataset unzipped successfully.")

def update_yaml(file_path):
    """
    Updates the YAML file with absolute paths for Colab and modifies the path key.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"data.yaml file not found at {file_path}")

    print("Updating data.yaml with absolute paths and path key...")
    # Read the existing YAML
    with open(file_path, "r") as yaml_file:
        existing_content = yaml_file.read()

    # Update the paths dynamically
    new_content = existing_content.replace("path: ../datasets/roboflow", "path: ..")
    new_content = new_content.replace("train: train/images", f"train: {os.path.join(PREPROCESSED_DATA_DIR, 'train/images')}")
    new_content = new_content.replace("val: valid/images", f"val: {os.path.join(PREPROCESSED_DATA_DIR, 'valid/images')}")
    new_content = new_content.replace("test: test/images", f"test: {os.path.join(PREPROCESSED_DATA_DIR, 'test/images')}")

    # Write the updated content back
    with open(file_path, "w") as yaml_file:
        yaml_file.write(new_content)

    print("data.yaml updated successfully.")

def main():
    """
    Main function to unzip and update the dataset YAML.
    """
    try:
        # Step 1: Unzip the dataset
        unzip_dataset(RAW_DATA_DIR, PREPROCESSED_DATA_DIR)

        # Step 2: Update the YAML file with correct paths
        update_yaml(DATA_YAML_PATH)

        print(f"Dataset preprocessing completed. Preprocessed data is available at: {PREPROCESSED_DATA_DIR}")
    except Exception as e:
        print(f"Error during dataset preprocessing: {e}")
        exit(1)

if __name__ == "__main__":
    main()