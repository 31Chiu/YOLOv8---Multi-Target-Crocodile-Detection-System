import os
import torch
import torch.nn as nn
import numpy as np
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ultralytics import YOLO
from tqdm import tqdm

# --- 1. Logging Configuration ---
# Configure logging to track the script's progress and for debugging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("yolo_svm_classifier.log"),
        logging.StreamHandler()
    ]
)

# --- 2. YOLOv8 Feature Extractor ---
class YOLOv8FeatureExtractor(nn.Module):
    """
    A module that wraps the YOLOv8 backbone for feature extraction.
    It loads a pre-trained YOLOv8 model, uses its backbone to generate feature maps,
    and then applies Global Average Pooling to convert them into 1D feature vectors.
    """
    def __init__(self, model_name='yolov8m.pt'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Load the full YOLO model
        yolo_model = YOLO(model_name)
        
        # Extract the backbone
        # We only use the first 10 layers (the backbone ending with SPPF).
        # This avoids the complex, non-sequential neck architecture that caused the error.
        self.feature_extractor = nn.Sequential(*list(yolo_model.model.model.children())[:10])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval() # Must be set to evaluation mode to disable layers like Dropout

        # Define the Global Average Pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    # def forward(self, x):
    #     """
    #     Forward pass function.
    #     Input: x (torch.Tensor) - A batch of images with shape (B, C, H, W)
    #     Output: torch.Tensor - Feature vectors with shape (B, feature_dim)
    #     """
    #     x = x.to(self.device)
        
    #     # Execute in a no-gradient context to save memory and computation
    #     with torch.no_grad():
    #         # Get the list of feature maps from the backbone
    #         # YOLOv8 outputs feature maps at multiple scales; we choose the last one
    #         # as it contains the highest-level semantic information.
    #         feature_maps = self.feature_extractor(x)
    #         last_feature_map = feature_maps[-1]
            
    #         # Apply Global Average Pooling, which turns a HxW feature map into 1x1
    #         pooled_features = self.pool(last_feature_map)
            
    #         # Flatten the features into a 2D tensor of shape (B, C)
    #         flattened_features = torch.flatten(pooled_features, 1)
            
    #     return flattened_features
    def forward(self, x):
        """Defines the forward pass."""
        # Move input data to the correct device (CPU or GPU)
        x = x.to(self.device)
        
        # Execute in a no-gradient context to save memory and computation
        with torch.no_grad():
            # 1. Extract feature maps directly with the backbone
            feature_maps = self.feature_extractor(x)
            
            # 2. Apply the pooling layer to the complete feature maps
            features = self.pool(feature_maps)
            
            # 3. Flatten the feature vector to a shape of (batch_size, channels)
            features = torch.flatten(features, 1)
            
        return features

# --- 3. Data Loading and Feature Extraction Helper Functions ---
def get_dataloader(data_dir, batch_size=32):
    """Creates and returns a DataLoader for the given directory."""
    # Use the same image transformation pipeline as in your reference scripts
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Using num_workers > 0 can speed up data loading
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def extract_features(dataloader, model):
    """Extracts all features and labels from a dataloader using the model."""
    features_list = []
    labels_list = []

    # Use tqdm to display a progress bar
    for images, labels in tqdm(dataloader, desc="Extracting Features"):
        feature_batch = model(images)
        features_list.append(feature_batch.cpu().numpy())
        labels_list.append(labels.numpy())
        
    # Concatenate the list of batches into a single large NumPy array
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

# --- 4. Main Execution Flow ---
def main():
    logging.info("Starting SVM classification using YOLOv8 features...")
    
    # Define dataset paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, './dataset/images/Training')
    val_dir = os.path.join(base_dir, './dataset/images/Validation')

    # Step 1: Initialize the feature extractor
    feature_extractor = YOLOv8FeatureExtractor(model_name='yolov8m.pt')

    # Step 2: Prepare the datasets
    logging.info("Loading training and validation datasets...")
    train_loader = get_dataloader(train_dir)
    val_loader = get_dataloader(val_dir)

    # Step 3: Extract features
    logging.info("Extracting features from the training set...")
    train_features, train_labels = extract_features(train_loader, feature_extractor)
    logging.info("Extracting features from the validation set...")
    val_features, val_labels = extract_features(val_loader, feature_extractor)
    logging.info(f"Feature extraction complete. Train features shape: {train_features.shape}, Validation features shape: {val_features.shape}")

    # Step 4: Train and evaluate the SVM classifier
    logging.info("\n--- Training and Evaluating SVM Classifier ---")
    
    # SVMs are sensitive to feature scaling, so standardizing the data is a crucial step.
    logging.info("Standardizing features...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    
    # Initialize the SVM. The 'linear' kernel is fast and often a good baseline.
    # 'rbf' is another common and powerful choice.
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    logging.info("Training the SVM classifier...")
    svm.fit(train_features_scaled, train_labels)

    logging.info("Making predictions on the validation set...")
    predictions = svm.predict(val_features_scaled)
    
    # Step 5: Calculate and display the final accuracy
    accuracy = accuracy_score(val_labels, predictions)
    logging.info("-" * 50)
    logging.info(f'FINAL RESULT: SVM classifier validation accuracy: {accuracy * 100:.2f}%')
    logging.info("-" * 50)

if __name__ == '__main__':
    main()