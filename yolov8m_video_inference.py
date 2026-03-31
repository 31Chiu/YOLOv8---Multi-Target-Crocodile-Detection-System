import os
import cv2
import time
import pandas as pd
from ultralytics import YOLO

# --- ⚙️ Core Configuration Area ---
# Ensure the path points to your best YOLOv8m weights
MODEL_WEIGHTS = 'My_YOLOv8_Runs/crocodile_detection_yolov8m/weights/best.pt'
INPUT_VIDEO_PATH = 'test_video.mp4'
OUTPUT_VIDEO_PATH = 'Output_YOLOv8m_BoTSORT.mp4'
CSV_OUTPUT_PATH = 'Log_YOLOv8m_BoTSORT.csv'
# -----------------------

def main():
    print("🚀 Starting YOLOv8m + BoT-SORT video dynamic tracking...")

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ Cannot find model weights file: {MODEL_WEIGHTS}")
        return
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"❌ Cannot find video file: {INPUT_VIDEO_PATH}")
        return

    # Load YOLOv8 model
    model = YOLO(MODEL_WEIGHTS)

    # Initialize video capture
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    csv_data = []
    frame_count = 0

    print("🎬 Starting frame-by-frame detection and tracking...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()

        # Core: Use .track() method to activate the tracker
        # persist=True ensures IDs are continuous between frames
        # tracker="botsort.yaml" calls the BoT-SORT algorithm
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
        
        # Extract the rendered result of the current frame (image with boxes and IDs)
        annotated_frame = results[0].plot()

        # Calculate FPS
        process_time = time.time() - start_time
        current_fps = 1.0 / process_time if process_time > 0 else 0

        # Parse tracking data and write to CSV
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            for track_id, box, conf, cls in zip(track_ids, boxes, confs, clss):
                x1, y1, x2, y2 = box
                csv_data.append({
                    'Frame': frame_count,
                    'Track_ID': track_id,
                    'Class_ID': cls,
                    'Confidence': round(conf, 4),
                    'BBox_X1': round(x1, 2),
                    'BBox_Y1': round(y1, 2),
                    'BBox_X2': round(x2, 2),
                    'BBox_Y2': round(y2, 2),
                    'FPS': round(current_fps, 2)
                })
        else:
            # If no targets are detected in this frame, record basic info
            csv_data.append({
                'Frame': frame_count,
                'Track_ID': None,
                'Class_ID': None,
                'Confidence': None,
                'BBox_X1': None, 'BBox_Y1': None, 'BBox_X2': None, 'BBox_Y2': None,
                'FPS': round(current_fps, 2)
            })

        # Display overall FPS info in the top left corner
        cv2.putText(annotated_frame, f"Model: YOLOv8m + BoT-SORT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        out_video.write(annotated_frame)

        if frame_count % 30 == 0:
            print(f"⏳ Progress: {frame_count}/{total_frames} frames processed...")

    cap.release()
    out_video.release()

    # Save log data
    df = pd.DataFrame(csv_data)
    df.to_csv(CSV_OUTPUT_PATH, index=False)

    print(f"\n🎉 Tracking complete!")
    print(f"📹 Rendered video saved to: {OUTPUT_VIDEO_PATH}")
    print(f"📊 Tracking trajectory data saved to: {CSV_OUTPUT_PATH}")

if __name__ == '__main__':
    main()