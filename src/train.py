import os
import random
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Set fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Paths
DATA_CONFIG_PATH = "data/preprocessed/data.yaml"
CONFIGS_DIR = "configs"
RESULTS_DIR = "results"
MODEL_SAVE_PATH = os.path.join(CONFIGS_DIR, "yolov8_pklot.pt")
EVALUATION_FILE = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")

# Ensure directories exist
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_model():
    """
    Trains a YOLOv8 model on the PKLot dataset.
    """
    print("Initializing YOLOv8 model...")
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Nano model for faster training; replace with 'yolov8m.pt' or others if needed

    print("Starting training...")
    # Train the model on the PKLot dataset
    model.train(
        data=DATA_CONFIG_PATH,  # Path to the dataset configuration
        epochs=20,              # Number of training epochs
        batch=16,               # Batch size
        imgsz=640,              # Input image size
        workers=4,              # Number of workers
        seed=SEED,              # Set the seed
        project=CONFIGS_DIR,    # Save configs and logs in the configs directory
        name="yolov8_pklot"     # Experiment name
    )
    print("Training completed!")

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return model

def evaluate_model(model):
    """
    Evaluates the trained YOLOv8 model and computes metrics.
    """
    print("Evaluating the model on the test set...")
    # Perform validation and testing on the test set
    results = model.val()

    # Extract predictions and ground truths for metric calculation
    y_true, y_pred = [], []
    for batch in results.pred:
        for pred in batch:  # Each prediction
            class_id = pred[-1].item()
            y_pred.append(int(class_id))
        for gt in batch:  # Each ground truth
            class_id = gt[-1].item()
            y_true.append(int(class_id))

    # Compute evaluation metrics
    precision = precision_score(y_true, y_pred, average="binary", pos_label=1)
    recall = recall_score(y_true, y_pred, average="binary", pos_label=1)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_pred)

    # Save metrics to a file
    with open(EVALUATION_FILE, "w") as f:
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred))

    print(f"Evaluation metrics saved to {EVALUATION_FILE}")

def main():
    """
    Main function to train and evaluate the YOLOv8 model.
    """
    try:
        # Step 1: Train the YOLOv8 model
        trained_model = train_model()

        # Step 2: Evaluate the trained model
        evaluate_model(trained_model)

        print("Model training and evaluation completed successfully!")
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        exit(1)

if __name__ == "__main__":
    main()