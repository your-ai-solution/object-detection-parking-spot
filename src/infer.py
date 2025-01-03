import os
import random
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Paths
MODEL_WEIGHTS_PATH = "configs/yolov8_pklot.pt"
TEST_IMAGES_PATH = "data/preprocessed/test/images"
TEST_LABELS_PATH = "data/preprocessed/test/labels"
RESULTS_DIR = "results"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_ground_truth(image, polygons, labels, title, save_path):
    """
    Plots the image with ground truth bounding boxes (polygonal) and saves it as a PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for i, polygon in enumerate(polygons):
        label = labels[i]
        color = 'red' if label == 1 else 'blue'  # Red for occupied, blue for empty
        polygon_points = [(polygon[j], polygon[j + 1]) for j in range(0, len(polygon), 2)]
        patch = patches.Polygon(polygon_points, closed=True, edgecolor=color, fill=False, linewidth=2)
        ax.add_patch(patch)
        ax.text(polygon_points[0][0], polygon_points[0][1] - 5, f"{'Occupied' if label == 1 else 'Empty'}",
                color=color, fontsize=10, weight='bold')

    ax.set_title(title)
    ax.axis('off')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_predictions(image, boxes, labels, title, save_path):
    """
    Plots the image with predicted bounding boxes (rectangular) and saves it as a PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label = labels[i]
        color = 'red' if label == 1 else 'blue'  # Red for occupied, blue for empty
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        ax.text(x1, y1 - 5, f"{'Occupied' if label == 1 else 'Empty'}", color=color, fontsize=10, weight='bold')

    ax.set_title(title)
    ax.axis('off')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def infer_and_visualize():
    """
    Loads the YOLO model, selects 5 random test images, and plots predictions and ground truth.
    """
    # Load the YOLO model
    print("Loading the YOLO model...")
    model = YOLO(MODEL_WEIGHTS_PATH)

    # Randomly select 5 images and their labels
    test_images = os.listdir(TEST_IMAGES_PATH)

    if len(test_images) < 5:
        print("Not enough images in the test folder.")
        return

    selected_images = random.sample(test_images, 5)

    for idx, random_image_file in enumerate(selected_images):
        image_path = os.path.join(TEST_IMAGES_PATH, random_image_file)
        label_path = os.path.join(TEST_LABELS_PATH, random_image_file.replace(".jpg", ".txt"))

        print(f"Processing image {idx + 1}: {random_image_file}")

        # Load the image
        image = cv2.imread(image_path)
        height, width, _ = image.shape  # Get image dimensions

        # Load ground truth labels
        ground_truth_polygons = []
        ground_truth_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                ground_truth = [line.strip().split() for line in f.readlines()]
            for label in ground_truth:
                try:
                    class_id = int(label[0])  # 0 = Empty, 1 = Occupied
                    polygon = [float(coord) for coord in label[1:]]
                    polygon = [int(coord * width) if i % 2 == 0 else int(coord * height)
                               for i, coord in enumerate(polygon)]
                    ground_truth_polygons.append(polygon)
                    ground_truth_labels.append(class_id)
                except ValueError:
                    print(f"Skipping invalid label: {label}")

        # Run inference with the model
        results = model(image_path)
        predictions = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        prediction_classes = results[0].boxes.cls.cpu().numpy()  # Get class IDs

        prediction_boxes = [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in predictions]
        prediction_labels = [int(cls) for cls in prediction_classes]

        # Plot and save results for ground truth and predictions
        plot_ground_truth(image, ground_truth_polygons, ground_truth_labels,
                          f"Ground Truth Detection - {random_image_file}",
                          os.path.join(RESULTS_DIR, f"ground_truth_{idx + 1}.png"))
        plot_predictions(image, prediction_boxes, prediction_labels,
                         f"Predicted Detection - {random_image_file}",
                         os.path.join(RESULTS_DIR, f"predicted_detection_{idx + 1}.png"))

    print("Inference and visualization for 5 images completed.")

def main():
    """
    Main function to perform inference and save visualizations.
    """
    try:
        infer_and_visualize()
    except Exception as e:
        print(f"Error during inference: {e}")
        exit(1)

if __name__ == "__main__":
    main()