

from ultralytics import YOLO
import os
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch


# Check the current working directory
print("Current working directory:", os.getcwd())

# Check if th config.yaml file exists
file_path = "project1/config.yaml"  # or "subdirectory/config.yaml"
print(f"Does {file_path} exist?", os.path.exists(file_path))

# If the file exists, train the model
if os.path.exists(file_path):
    results = model.train(data=file_path, epochs=100)


