import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. YOLOv8 Wrapper for EigenCAM ---
class YOLOv8Wrapper(torch.nn.Module):
    def __init__(self, yolov8_model):
        super().__init__()
        self.model = yolov8_model.model 
        
    def forward(self, x):
        preds = self.model(x)
        scores = preds[0][:, 4:, :] 
        return scores.max(dim=-1)[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting YOLOv8m (BoT-SORT + EigenCAM) using device: {device}")

    # --- ⚙️ Core Configuration Area ---
    MODEL_WEIGHTS = 'My_YOLOv8_Runs/crocodile_detection_yolov8m/weights/best.pt'
    INPUT_VIDEO_PATH = 'test_video.mp4'
    OUTPUT_VIDEO_PATH = 'Output_YOLOv8m_Track_and_CAM.mp4'
    CSV_OUTPUT_PATH = 'Log_YOLOv8m_Track_and_CAM.csv'
    # -----------------------

    if not os.path.exists(MODEL_WEIGHTS) or not os.path.exists(INPUT_VIDEO_PATH):
        print("❌ Error: Cannot find model weights or video file.")
        return

    # 1. Load Model & Initialize CAM
    base_model = YOLO(MODEL_WEIGHTS)
    model_wrapper = YOLOv8Wrapper(base_model).to(device).eval()
    
    # Target the 21st layer of YOLOv8m backbone
    target_layers = [base_model.model.model[21]]
    cam = EigenCAM(model=model_wrapper, target_layers=target_layers)

    # 2. Initialize Video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    csv_data = []
    frame_count = 0
    print("🎬 Starting frame-by-frame dual-track processing...")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            start_time = time.time()

            # --- TRACKING TRACK: Get Bounding Boxes & IDs ---
            # tracker="botsort.yaml" is activated here
            results = base_model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
            
            # --- EXPLAINABILITY TRACK: Generate EigenCAM Heatmap ---
            img_size = 640
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_img_resized = cv2.resize(rgb_img, (img_size, img_size))
            rgb_img_float = np.float32(rgb_img_resized) / 255.0
            input_tensor = torch.from_numpy(rgb_img_float).permute(2, 0, 1).unsqueeze(0).to(device)

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
            
            # Resize heatmap back to original video resolution
            final_frame = cv2.resize(visualization, (width, height))
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)

            # --- VISUAL FUSION: Draw Tracking Data over Heatmap ---
            detected_count = 0
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                boxes = results[0].boxes.xyxy.cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
                clss = results[0].boxes.cls.int().cpu().tolist()
                detected_count = len(track_ids)

                for track_id, box, conf, cls in zip(track_ids, boxes, confs, clss):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Log data to CSV
                    csv_data.append({
                        'Frame': frame_count, 'Track_ID': track_id, 'Class_ID': cls,
                        'Confidence': round(conf, 4), 'BBox_X1': x1, 'BBox_Y1': y1, 
                        'BBox_X2': x2, 'BBox_Y2': y2
                    })

                    # Draw crisp Green Bounding Box and ID on top of the heatmap
                    cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{track_id} {conf:.2f}"
                    cv2.putText(final_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                csv_data.append({
                    'Frame': frame_count, 'Track_ID': None, 'Class_ID': None,
                    'Confidence': None, 'BBox_X1': None, 'BBox_Y1': None, 
                    'BBox_X2': None, 'BBox_Y2': None
                })

            # Calculate FPS and draw global info
            process_time = time.time() - start_time
            current_fps = 1.0 / process_time if process_time > 0 else 0
            # Update CSV with FPS for this frame
            csv_data[-1]['FPS'] = round(current_fps, 2) if len(csv_data) > 0 else 0

            cv2.putText(final_frame, f"Model: YOLOv8m (BoT-SORT + EigenCAM)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(final_frame, f"Crocs Tracked: {detected_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if detected_count > 0 else (0, 0, 255), 2)
            cv2.putText(final_frame, f"FPS: {current_fps:.1f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            out_video.write(final_frame)
            
            if frame_count % 10 == 0:
                print(f"⏳ Progress: {frame_count}/{total_frames} frames... (FPS: {current_fps:.1f})")

    cap.release()
    out_video.release()
    
    df = pd.DataFrame(csv_data)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\n🎉 Processing complete! Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == '__main__':
    main()