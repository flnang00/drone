import cv2
import os
import tellopy
import numpy as np
from ultralytics import YOLO

def video_handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_VIDEO_FRAME:
        image = cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)

        # Run inference on the frame
        results = model(image)

        # Render the results on the frame
        results.render()  # This will add bounding box and label overlays

        # Display the frame
        cv2.imshow('Live Stream', image)

        # Check for keyboard input to quit (press 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            drone.quit()

model_path = os.path.join(r"C:\Users\Florian Nartea\Desktop\Cigarette_3\weights\last.pt")

# Load a model
model = YOLO(model_path)  # load a custom model

# Connect to the Tello drone
drone = tellopy.Tello()

drone.subscribe(drone.EVENT_VIDEO_FRAME, video_handler)

drone.connect()
drone.wait_for_connection(60.0)

try:
    drone.set_video_mode(True)  # Enable video mode
    while True:
        pass
except Exception as ex:
    print(ex)
finally:
    drone.quit()
    cv2.destroyAllWindows()