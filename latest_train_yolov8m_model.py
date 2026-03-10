import os
import torch
import logging
from ultralytics import YOLO

# Configure logging to record training progress in the console and a file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('latest_training_yolov8m.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class YOLOv8Trainer:
    def __init__(self, model_name='yolov8m.pt', data_config='data.yaml', epochs=50, batch_size=32, learning_rate=0.001, project_name='My_YOLOv8_Runs', experiment_name='crocodile_detection_yolov8m'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            torch.cuda.set_device(0) 
            logging.info("CUDA is available. Using GPU.")
        else:
            logging.info("CUDA not available. Using CPU.")

        self.model_name = model_name
        self.data_config = data_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.results_dir = os.path.join(self.project_name, self.experiment_name)
        self.model = None

    def _build_and_load_model(self, weights_path):
        logging.info(f"Loading model with weights: {weights_path}")
        model = YOLO(weights_path)
        model.to(self.device)
        return model

    # --- 新增：自定义回调函数，提取 YOLO 内部数据并复刻 ResNet 日志格式 ---
    def _log_epoch_metrics(self, trainer):
        epoch = trainer.epoch + 1
        num_epochs = trainer.epochs
        lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0.0
        
        # 提取训练损失 (YOLOv8 默认包含 box_loss 和 cls_loss)
        if trainer.tloss is not None:
            box_loss = trainer.tloss[0].item()
            cls_loss = trainer.tloss[1].item()
        else:
            box_loss, cls_loss = 0.0, 0.0
            
        # 提取验证指标 (YOLOv8 会在 trainer.metrics 中保存验证结果)
        metrics = trainer.metrics or {}
        prec = metrics.get('metrics/precision(B)', 0.0)
        rec = metrics.get('metrics/recall(B)', 0.0)
        map50 = metrics.get('metrics/mAP50(B)', 0.0)
        
        # 按照之前 ResNet 的格式进行日志输出
        logging.info(
            f"Epoch {epoch:02d}/{num_epochs} [LR: {lr:.6f}] | "
            f"Train -> Box Loss: {box_loss:.4f} Cls Loss: {cls_loss:.4f} | "
            f"Val -> Prec: {prec:.4f} Rec: {rec:.4f} mAP50: {map50:.4f}"
        )

    def train(self):
        try:
            self.model = self._build_and_load_model(self.model_name)
            
            # --- 新增：将我们写好的回调函数绑定到 YOLO 引擎上 ---
            self.model.add_callback("on_fit_epoch_end", self._log_epoch_metrics)
            
            logging.info("Starting YOLOv8 training with strict data augmentation and custom logging...")
            
            self.model.train(
                data=self.data_config,
                epochs=self.epochs,
                batch=self.batch_size,
                lr0=self.learning_rate,
                project=self.project_name,
                name=self.experiment_name,
                imgsz=640,
                patience=10,
                exist_ok=True,
                
                # --- Preserve stringent data augmentation parameters (align with ResNet) ---
                degrees=30.0,       # Corresponds to RandomRotation(30)
                translate=0.1,      # The corresponding translation is translate=(0.1, 0.1).
                scale=0.1,          # Corresponding scaling
                shear=10.0,         # The corresponding shear=10 for RandomAffine
                # --- The error-causing `erasing=0.2` has been removed, and its occlusion effect 
                # has been replaced by YOLO's default Mosaic stitching. ---
                hsv_h=0.1,          # Hue dithering
                hsv_s=0.4,          # Saturation jitter
                hsv_v=0.4           # Brightness jitter
            )

            logging.info("Training complete.")
            self.evaluate_best_model()

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise

    def evaluate_best_model(self):
        try:
            best_model_path = os.path.join(self.results_dir, 'weights', 'best.pt')
            
            if not os.path.exists(best_model_path):
                logging.error(f"Best model not found at path: {best_model_path}")
                return

            logging.info("-" * 50)
            best_model = self._build_and_load_model(best_model_path)
            metrics = best_model.val(data=self.data_config)
            
            map50_95 = metrics.box.map
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
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    trainer = YOLOv8Trainer()
    trainer.train()

if __name__ == '__main__':
    main()