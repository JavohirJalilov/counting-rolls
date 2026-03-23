import cv2
from ultralytics import YOLO
import os

def run_inference(model_path, video_path, output_path):
    # Load the model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO inference on the frame
        results = model(frame, verbose=False)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Write the annotated frame to the output video
        out.write(annotated_frame)
        
    # Release everything if job is finished
    cap.release()
    out.release()
    print(f"Inference complete. Saved to {output_path}")

if __name__ == "__main__":
    model_path = "weights.pt"
    video_path = "videos/rolls_cut.mp4"
    output_path = "videos/rolls_cut_annotated.mp4"
    
    if not os.path.exists("videos"):
        os.makedirs("videos")
        
    run_inference(model_path, video_path, output_path)
