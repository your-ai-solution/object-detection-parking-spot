import os
import random
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Paths
MODEL_WEIGHTS_PATH = "configs/yolov8_pklot.pt"
DATA_CONFIG_PATH = "data/preprocessed/data.yaml"
RESULTS_DIR = "results"
EVALUATION_FILE = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_model():
    """
    Evaluates the YOLOv8 model on the test dataset and saves metrics to the results directory.
    """
    print("Loading the YOLO model...")
    # Load the trained YOLOv8 model
    model = YOLO(MODEL_WEIGHTS_PATH)

    print("Evaluating the model on the test set...")
    # Perform evaluation on the test dataset
    results = model.val(
        data=DATA_CONFIG_PATH,  # Dataset configuration
        split="test",           # Explicitly specify the test split
        save=True,              # Save detection results
        project=RESULTS_DIR,    # Save results directly to the evaluation directory
        name="evaluation"       # Save directly into the specified folder
    )

    # Extract metrics
    precision = results.box.p  # Precision for each class
    recall = results.box.r     # Recall for each class
    map50 = results.box.map50  # mAP at IoU threshold 0.5
    map50_95 = results.box.map  # mAP from IoU 0.5 to 0.95

    # Save metrics to a file
    with open(EVALUATION_FILE, "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"Precision: {precision.mean():.4f}\n")
        f.write(f"Recall: {recall.mean():.4f}\n")
        f.write(f"mAP@0.5: {map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map50_95:.4f}\n")
        f.write(f"\nDetailed evaluation results are saved in {RESULTS_DIR}\n")

    print(f"Evaluation metrics saved to {EVALUATION_FILE}")
    print("Evaluation completed successfully!")

def main():
    """
    Main function to evaluate the model.
    """
    try:
        evaluate_model()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        exit(1)

if __name__ == "__main__":
    main()