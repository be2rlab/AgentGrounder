import os
from ultralytics import YOLOE

root_dir = os.getenv("BASE_DIR") or "./"

# 1. Initialize the YOLOE model
# Options include: yoloe-11s-seg.pt (small), yoloe-11m-seg.pt (medium), yoloe-v8s-seg.pt, etc.
model = YOLOE(os.path.join(root_dir, "weights/ultralytics/yoloe-26x-seg.pt"))

# 2. Define your custom open-vocabulary classes
# You can type any objects or descriptive text you want to detect!
custom_classes = ["sofa", "chair", "door", "painting", "human", 'door', 'monitor', 'pillow', 'floor', 'wall']

# Encode the text prompts into embeddings and set them as the vocabulary
model.set_classes(custom_classes)

# 3. Perform inference on an image
# You can use a local file path, a video, or an image URL.
# Adjusting the 'conf' (confidence threshold) helps catch less obvious objects.
filename = "rendered"
results = model.predict(source=os.path.join(root_dir, "figs/{filename}.png"), conf=0.1)

# 4. View and save the results
for result in results:
    result.show()  # Opens a window displaying the image with bounding boxes AND segmentation masks
    result.save(filename=os.path.join(root_dir, "examples/output/{filename}_yoloe_rendered.png"))  # Saves the annotated image to your disk