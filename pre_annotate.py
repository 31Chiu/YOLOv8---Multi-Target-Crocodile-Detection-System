import os
import glob
import cv2
from ultralytics import YOLO
import logging

# --- Configuration Section ---
# Model name: We choose a large, high-accuracy pre-trained model to maximize the likelihood of detecting objects.
MODEL_NAME = 'yolov8x.pt'  # 'x' for extra-large model

# Root directory of your crocodile dataset
DATASET_BASE_DIR = './dataset'

# Confidence threshold: Only save predictions with a confidence score higher than this value.
CONFIDENCE_THRESHOLD = 0.25

# Class ID for 'crocodile' in your data.yaml (since we have only one class, it's 0)
CROCODILE_CLASS_ID = 0
# --- End of Configuration Section ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_paths(directory):
    """Gets the paths of all image files in the specified directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
    return image_paths

def convert_to_yolo_format(box, img_width, img_height):
    """
    Converts the model's bounding box output (xyxy format) to YOLO's .txt annotation format (normalized xywh).
    xyxy: [x_min, y_min, x_max, y_max]
    xywh: [x_center_norm, y_center_norm, width_norm, height_norm]
    """
    # Extract coordinates from the box object
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # Calculate center point coordinates and width/height
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    # Normalize by dividing by the image's width and height
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return f"{CROCODILE_CLASS_ID} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

def annotate_directory(image_dir, label_dir, model):
    """Performs pre-annotation for all images in the specified directory."""
    logging.info(f"Starting to process directory: {image_dir}")
    
    # Ensure the label output directory exists
    os.makedirs(label_dir, exist_ok=True)
    
    image_paths = get_image_paths(image_dir)
    if not image_paths:
        logging.warning(f"No images found in directory: {image_dir}")
        return

    for img_path in image_paths:
        # Read the image to get its dimensions
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape

        # Use the model for prediction
        results = model.predict(source=img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]  # Get the results for the first image

        # Prepare to write the annotation file
        label_filename = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        with open(label_path, 'w') as f:
            if result.boxes:
                # Iterate through all detected bounding boxes
                for box in result.boxes:
                    yolo_line = convert_to_yolo_format(box, img_width, img_height)
                    f.write(yolo_line + '\n')
        
        logging.info(f"Annotation file generated: {label_path}")

    logging.info(f"Finished processing directory: {image_dir}")

def main():
    """Main function to load the model and start the annotation process."""
    logging.info("Starting the automated pre-annotation process...")
    
    # 1. Load the pre-trained YOLOv8 model
    try:
        model = YOLO(MODEL_NAME)
        logging.info(f"Model loaded successfully: {MODEL_NAME}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # 2. Define image and label folder paths
    train_images_dir = os.path.join(DATASET_BASE_DIR, 'images/Training')
    val_images_dir = os.path.join(DATASET_BASE_DIR, 'images/Validation')

    train_labels_dir = os.path.join(DATASET_BASE_DIR, 'labels/Training')
    val_labels_dir = os.path.join(DATASET_BASE_DIR, 'labels/Validation')

    # 3. Process the training and validation sets separately
    annotate_directory(train_images_dir, train_labels_dir, model)
    annotate_directory(val_images_dir, val_labels_dir, model)
    
    logging.info("Automated pre-annotation process completed!")
    logging.warning("Important: Be sure to manually check and correct all generated .txt annotation files!")

if __name__ == '__main__':
    main()