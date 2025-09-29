import cv2
from ultralytics import YOLO
import numpy as np

# --- Configuration ---
# Change this path to the actual path of your trained best.pt model
MODEL_PATH = 'My_YOLOv8_Runs/crocodile_detection_yolov8m/weights/best.pt'
# Set a confidence threshold (e.g., 0.7 means 70% confidence)
CONFIDENCE_THRESHOLD = 0.7
# --- End of Configuration ---

def main():
    # 1. Load the Model
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")

    # 2. Open the Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    print("Camera opened. Press the 'q' key to close the window.")

    # 3. Main Loop
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera.")
            break
            
        # --- Object Counting Feature: Reset counter for each frame ---
        object_count = 0

        # Use the model to predict objects in the current frame
        results = model(frame, verbose=False)

        # Loop through each detected object
        for box in results[0].boxes:
            # Check if the confidence is above our threshold
            if box.conf[0] > CONFIDENCE_THRESHOLD:
                # --- Object Counting Feature: Increment counter if detection is confident ---
                object_count += 1

                # Get coordinates of the bounding box
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                
                # Get the confidence score and class name
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Draw the rectangle on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create the label text (e.g., "crocodile: 0.95")
                label = f"{class_name}: {confidence:.2f}"
                
                # Put the label text above the rectangle
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Object Counting Feature: Display the total count on the frame ---
        count_text = f"Objects Detected: {object_count}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the processed frame
        cv2.imshow("YOLOv8 Real-Time Inference", frame)

        # Exit condition: press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 4. Release Resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program finished.")

if __name__ == "__main__":
    main()