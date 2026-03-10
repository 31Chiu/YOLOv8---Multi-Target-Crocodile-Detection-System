import os
import glob
import cv2
from ultralytics import YOLO
import logging

# --- Configuration Section ---
MODEL_NAME = 'yolov8x.pt'  
DATASET_BASE_DIR = './dataset'
CONFIDENCE_THRESHOLD = 0.5
CROCODILE_CLASS_ID = 0

# 新增配置：定义背景图片所在的文件夹名称
BACKGROUND_FOLDER_NAME = 'non-crocodile'
# --- End of Configuration Section ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_paths(directory):
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
    return image_paths

def convert_to_yolo_format(box, img_width, img_height):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return f"{CROCODILE_CLASS_ID} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

# 修改点 1：接收 sub_dir_name 参数
def annotate_directory(image_dir, label_dir, model, sub_dir_name=""):
    logging.info(f"Starting to process directory: {image_dir}")
    os.makedirs(label_dir, exist_ok=True)
    
    image_paths = get_image_paths(image_dir)
    if not image_paths:
        logging.warning(f"No images found in directory: {image_dir}")
        return

    # 检查当前是否为背景文件夹
    is_background = (sub_dir_name.lower() == BACKGROUND_FOLDER_NAME.lower())
    if is_background:
        logging.info(f"Detected background folder '{sub_dir_name}'. Generating empty annotations to save time.")

    for img_path in image_paths:
        label_filename = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        # 以写入模式打开文件
        with open(label_path, 'w') as f:
            if not is_background:
                # 修改点 2：只有在“不是背景文件夹”时，才加载图片并运行模型预测
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"Could not read image, skipping: {img_path}")
                    continue
                img_height, img_width, _ = img.shape

                results = model.predict(source=img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
                result = results[0] 

                if result.boxes:
                    for box in result.boxes:
                        yolo_line = convert_to_yolo_format(box, img_width, img_height)
                        f.write(yolo_line + '\n')
        
    logging.info(f"Finished processing directory: {image_dir}")

def process_base_directory(base_image_dir, base_label_dir, model):
    all_image_paths = []
    if not os.path.isdir(base_image_dir):
        logging.warning(f"Base image directory not found, skipping: {base_image_dir}")
        return all_image_paths

    sub_dirs = [d for d in os.listdir(base_image_dir) if os.path.isdir(os.path.join(base_image_dir, d))]

    if not sub_dirs:
        logging.info(f"No subdirectories found in '{base_image_dir}'. Processing images directly.")
        annotate_directory(base_image_dir, base_label_dir, model)
        all_image_paths.extend(get_image_paths(base_image_dir))
    else:
        logging.info(f"Found subdirectories in '{base_image_dir}': {sub_dirs}. Processing each one.")
        for sub_dir in sub_dirs:
            image_dir = os.path.join(base_image_dir, sub_dir)
            label_dir = os.path.join(base_label_dir, sub_dir)
            
            # 修改点 3：将 sub_dir 传递给 annotate_directory
            annotate_directory(image_dir, label_dir, model, sub_dir)
            all_image_paths.extend(get_image_paths(image_dir))
            
    return all_image_paths

def main():
    logging.info("Starting the automated pre-annotation process...")
    
    try:
        model = YOLO(MODEL_NAME)
        logging.info(f"Model loaded successfully: {MODEL_NAME}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    train_images_base_dir = os.path.join(DATASET_BASE_DIR, 'images/Training')
    val_images_base_dir = os.path.join(DATASET_BASE_DIR, 'images/Validation')
    train_labels_base_dir = os.path.join(DATASET_BASE_DIR, 'labels/Training')
    val_labels_base_dir = os.path.join(DATASET_BASE_DIR, 'labels/Validation')

    train_image_paths = process_base_directory(train_images_base_dir, train_labels_base_dir, model)
    val_image_paths = process_base_directory(val_images_base_dir, val_labels_base_dir, model)

    if train_image_paths:
        cvat_manifest_path = 'train.txt'
        with open(cvat_manifest_path, 'w', encoding='utf-8') as f:
            for img_path in train_image_paths:
                filename = os.path.basename(img_path)
                f.write(f"data/obj_train_data/{filename}\n")
        logging.info(f"CVAT manifest successfully generated: {cvat_manifest_path} ({len(train_image_paths)} lines)")

    if val_image_paths:
        with open('val.txt', 'w', encoding='utf-8') as f:
            for img_path in val_image_paths:
                filename = os.path.basename(img_path)
                f.write(f"data/obj_train_data/{filename}\n")
        logging.info(f"CVAT validation manifest generated: val.txt ({len(val_image_paths)} lines)")
    
    logging.info("Automated pre-annotation process completed!")

if __name__ == '__main__':
    main()