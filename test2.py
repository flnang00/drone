import cv2
import os
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn import visualize


class InferenceConfig(Config):
    # Set configuration values here
    NAME = "cig_butts"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + cigarette

config = InferenceConfig()
MODEL_DIR = os.getcwd()  # Use current working directory

# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

# Load weights
weights_path = r"C:\Users\Florian Nartea\Desktop\Cigarette_3\weights\mask_rcnn_cig_butts_0008.h5"
model.load_weights(weights_path, by_name=True)

class_names = ['BG', 'cigarette']

# Open video file
video_path = r"C:\Users\Florian Nartea\Desktop\Cigarette_3\1099977541-preview.mp4"


import os

if os.path.exists(video_path):
    print("Video file exists")
else:
    print("Video file does not exist")
    
    
    
    