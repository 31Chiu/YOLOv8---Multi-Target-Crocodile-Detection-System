import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM # 🔄 核心替换 1：导入 EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. 定义 YOLOv8 的翻译器 (Wrapper) ---
class YOLOv8Wrapper(torch.nn.Module):
    def __init__(self, yolov8_model):
        super().__init__()
        self.model = yolov8_model.model 
        
    def forward(self, x):
        preds = self.model(x)
        scores = preds[0][:, 4:, :] 
        return scores.max(dim=-1)[0]

def main():
    image_name = input("请输入要测试的图片名称 (例如 sample1.jpg): ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {device}")

    # --- 2. 加载你训练好的 YOLOv8m 模型 ---
    yolo_weights_path = 'My_YOLOv8_Runs/crocodile_detection_yolov8m/weights/best.pt' 
    if not os.path.exists(yolo_weights_path):
        print(f"错误: 找不到 YOLO 权重文件 '{yolo_weights_path}'")
        return

    base_model = YOLO(yolo_weights_path)

    # 🚫 注意：这里我们彻底删除了“手动解冻梯度”的 for 循环代码！
    
    model = YOLOv8Wrapper(base_model).to(device).eval()

    # --- 3. 锁定目标层 ---
    target_layers = [base_model.model.model[21]]

    # --- 4. 图像预处理 ---
    input_dir = 'Test_Eigen-CAM_Images'
    img_path = os.path.join(input_dir, image_name) 
    if not os.path.exists(img_path):
        print(f"错误: 找不到图片 '{img_path}'")
        return

    img_size = 640
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (img_size, img_size))
    rgb_img_float = np.float32(rgb_img) / 255
    
    input_tensor = torch.from_numpy(rgb_img_float).permute(2, 0, 1).unsqueeze(0).to(device)

    # --- 5. 生成热力图 ---
    # 🔄 核心替换 2：使用 EigenCAM 类
    cam = EigenCAM(model=model, target_layers=target_layers)
    
    # 🔄 核心替换 3：EigenCAM 依赖主成分分析，不需要像 Grad-CAM 那样指定特定的类别 Target
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # --- 6. 绘制并保存结果 ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image (640x640)")
    plt.imshow(rgb_img_float)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("YOLOv8m EigenCAM")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    
    output_dir = 'Test_Eigen-CAM_Results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"YOLOv8m_EigenCAM_{image_name}")
    plt.savefig(output_path)
    print(f"分析完成！YOLOv8m EigenCAM 热力图已保存至 {output_path}")

if __name__ == '__main__':
    main()