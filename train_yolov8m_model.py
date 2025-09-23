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
    A class that encapsulates the YOLOv8 training process, inspired by the structure of train_cnn_model.py.
    It centralizes all training-related settings and operations, making the main program logic clear.
    """
    def __init__(self, model_name='yolov8m.pt', data_config='data.yaml', epochs=50, batch_size=32, learning_rate=0.001, project_name='runs/detect', experiment_name='crocodile_experiment'):
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

        # 3. Set output directories
        self.project_name = project_name
        self.experiment_name = experiment_name

        # 4. Build the AI model
        self.model = self._build_model()

    def _build_model(self):
        """
        Loads the pre-trained YOLOv8 model.
        This is the main difference from the original CNN script, which built a model from scratch.
        Here, we leverage the concept of transfer learning by fine-tuning a powerful pre-trained model.
        """
        logging.info(f"Loading pre-trained model: {self.model_name}")
        # Load a pre-trained YOLOv8 model
        model = YOLO(self.model_name)
        # Move the model to the specified device (although ultralytics handles this automatically, being explicit is good practice)
        model.to(self.device)
        return model

    def train(self):
        """
        The main function to start the YOLOv8 training process.
        This function corresponds to the `train` method in the original CNN script, but the `ultralytics`
        library encapsulates the complex training loop (train_epoch, validate) into a single, simple .train() call.
        """
        try:
            logging.info("Starting YOLOv8 training...")
            logging.info(f"  - Model: {self.model_name}")
            logging.info(f"  - Data Config: {self.data_config}")
            logging.info(f"  - Epochs: {self.epochs}")
            logging.info(f"  - Batch Size: {self.batch_size}")
            logging.info(f"  - Learning Rate: {self.learning_rate}")

            # Call the ultralytics train function
            self.model.train(
                data=self.data_config,
                epochs=self.epochs,
                batch=self.batch_size,
                lr0=self.learning_rate,  # lr0 is the initial learning rate
                project=self.project_name,
                name=self.experiment_name,
                imgsz=640,  # Image size, 640 is a common default
                patience=10, # Early stopping patience, stops if no improvement after 10 epochs
                exist_ok=True # Do not raise an error if the experiment directory already exists
            )

            logging.info("Training complete.")
            logging.info(f"Results saved to: {os.path.join(self.project_name, self.experiment_name)}")

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
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
    # All parameters are fixed here, aligning with your "fixed configuration file" requirement
    trainer = YOLOv8Trainer(
        model_name='yolov8m.pt',
        data_config='data.yaml',
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        project_name='My_YOLOv8_Runs',
        experiment_name='crocodile_detection_yolov8m'
    )
    trainer.train()

if __name__ == '__main__':
    main()