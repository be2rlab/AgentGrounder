import os
from ultralytics import YOLOWorld

root_dir = os.getenv("BASE_DIR") or "./"

# 1. Initialize the YOLO-World model
# Options include: yolov8s-worldv2.pt (small), yolov8m-worldv2.pt (medium), etc.
model = YOLOWorld(os.path.join(root_dir, "weights/ultralytics/yolov8x-worldv2.pt"))

# 2. Define your custom open-vocabulary classes
# You can type any objects or descriptive text you want to detect!
custom_classes = ["sofa", "chair", "door", "painting", "human", "snack machine"]
model.set_classes(custom_classes)

# 3. Perform inference on an image
# You can use a local file path, a video, or an image URL.
# Adjusting the 'conf' (confidence threshold) helps catch less obvious objects.
filename = "object_4"
results = model.predict(source=os.path.join(root_dir, f"figs/{filename}.jpg"), conf=0.1)

# 4. View and save the results
for result in results:
    result.show()  # Opens a window displaying the image with bounding boxes
    result.save(filename=os.path.join(root_dir, f"examples/output/{filename}_yolo_world_results.jpg"))  # Saves the annotated image to your disk