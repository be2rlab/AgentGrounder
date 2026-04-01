import os
from ultralytics.models.sam import SAM3SemanticPredictor

root_dir = os.getenv("BASE_DIR") or "./"

# Initialize predictor (save=True enables output saving)
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=os.path.join(root_dir, "weights/sam3/sam3.pt"),  # Path to your sam3.pt file
    half=True,  # FP16 for speed
    save=True,  # Saves annotated image to runs/segment/predict/
    save_dir=os.path.join(root_dir, "examples/output")
)

predictor = SAM3SemanticPredictor(overrides=overrides)

# Load image and predict with text prompt (segments all "person" instances)
filename = "rendered"
predictor.set_image(os.path.join(root_dir, f"figs/{filename}.png"))  # Replace with your image path
results = predictor(text=["man"])

print(results[0])  # View results summary
