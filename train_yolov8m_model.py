import os
import torch
import logging
from ultralytics import YOLO

# Configure logging to record training progress in the console and a file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_yolo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class YOLOv8Trainer:
    """
    A class that encapsulates the YOLOv8 training and evaluation process.
    """
    def __init__(self, model_name='yolov8m.pt', data_config='data.yaml', epochs=50, batch_size=32, learning_rate=0.001, project_name='My_YOLOv8_Runs', experiment_name='crocodile_detection_yolov8m'):
        """
        Initializes the trainer and sets up all key parameters.
        """
        # 1. Set the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            torch.cuda.set_device(0) # Explicitly use the first GPU
            logging.info("CUDA is available. Using GPU.")
        else:
            logging.info("CUDA not available. Using CPU.")

        # 2. Define training parameters
        self.model_name = model_name
        self.data_config = data_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 3. Set output directories and construct the path for the results
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.results_dir = os.path.join(self.project_name, self.experiment_name)

        # 4. Initialize model placeholder
        self.model = None

    def _build_and_load_model(self, weights_path):
        """
        Builds the YOLO model and loads specified weights.
        """
        logging.info(f"Loading model with weights: {weights_path}")
        model = YOLO(weights_path)
        model.to(self.device)
        return model

    def train(self):
        """
        The main function to start the YOLOv8 training process.
        """
        try:
            # Load the initial pre-trained model for training
            self.model = self._build_and_load_model(self.model_name)
            
            logging.info("Starting YOLOv8 training...")
            logging.info(f"  - Model: {self.model_name}")
            logging.info(f"  - Data Config: {self.data_config}")
            logging.info(f"  - Epochs: {self.epochs}")
            logging.info(f"  - Batch Size: {self.batch_size}")

            # Call the ultralytics train function
            self.model.train(
                data=self.data_config,
                epochs=self.epochs,
                batch=self.batch_size,
                lr0=self.learning_rate,
                project=self.project_name,
                name=self.experiment_name,
                imgsz=640,
                patience=10,
                exist_ok=True
            )

            logging.info("Training complete.")
            logging.info(f"Results saved to: {self.results_dir}")

            # After training, automatically evaluate the best model
            self.evaluate_best_model()

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise

    def evaluate_best_model(self):
        """
        Loads the best performing model weights and evaluates it on the validation set.
        """
        try:
            best_model_path = os.path.join(self.results_dir, 'weights', 'best.pt')
            
            if not os.path.exists(best_model_path):
                logging.error(f"Best model not found at path: {best_model_path}")
                return

            logging.info("-" * 50)
            logging.info("Starting evaluation of the best model...")
            logging.info(f"Loading best model from: {best_model_path}")

            # Load the best model
            best_model = self._build_and_load_model(best_model_path)
            
            # Evaluate the model on the validation set
            # The .val() method returns a results object with all performance metrics
            metrics = best_model.val(data=self.data_config)
            
            logging.info("Evaluation complete.")
            logging.info("-" * 50)
            logging.info("Performance Metrics of the Best Model:")
            
            # Extract and log the key metrics
            # mAP50-95 is the primary metric for object detection accuracy
            map50_95 = metrics.box.map
            # mAP50 is the accuracy at an IoU threshold of 50%
            map50 = metrics.box.map50
            precision = metrics.box.mp
            recall = metrics.box.mr

            logging.info(f"  - mAP50-95 (primary metric): {map50_95:.4f}")
            logging.info(f"  - mAP50: {map50:.4f}")
            logging.info(f"  - Precision: {precision:.4f}")
            logging.info(f"  - Recall: {recall:.4f}")
            logging.info("-" * 50)

        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")
            raise

def main():
    """
    Main function to configure and run the trainer.
    """
    # Set a random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize and run the trainer
    trainer = YOLOv8Trainer(
        model_name='yolov8m.pt',
        data_config='data.yaml',
        epochs=50,
        batch_size=32, # Reduced batch size for stability, adjust if you have a high-end GPU
        learning_rate=0.001,
        project_name='My_YOLOv8_Runs',
        experiment_name='crocodile_detection_yolov8m'
    )
    trainer.train()

if __name__ == '__main__':
    main()