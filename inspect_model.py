from ultralytics import YOLO

def inspect_model(model_path):
    try:
        model = YOLO(model_path)
        print(f"Model: {model_path}")
        print(f"Number of classes: {len(model.names)}")
        print(f"Class names: {model.names}")
    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    inspect_model("weights.pt")
