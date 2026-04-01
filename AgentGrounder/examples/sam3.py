import torch
#################################### For Image ####################################
from PIL import Image
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model(
    bpe_path="../sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    device="cuda",
    eval_mode=True,
    checkpoint_path="weights/sam3/sam3.pt",
    load_from_HF=False,
)
processor = Sam3Processor(model)
# Load an image
image = Image.open("figs/rendered.png")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="man")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

masks = masks.cpu()
boxes = boxes.cpu()
scores = scores.cpu()
import numpy as np

def draw_and_save_sam3_results(image, masks, boxes, scores, save_path='sam3_results.png'):
    """
    Draws the original image, segmentation masks, and bounding boxes with scores,
    then saves the combined visualization to a file.

    Args:
        image (PIL.Image.Image): The original image.
        masks (torch.Tensor): Tensor of segmentation masks.
        boxes (torch.Tensor): Tensor of bounding boxes.
        scores (torch.Tensor): Tensor of scores for each mask/box.
        save_path (str): The path to save the resulting image.
    """
    image_np = np.array(image)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(image_np)
    for idx, mask in enumerate(masks[:5]):  # Limiting to first 5 for clarity
        mask_np = mask.squeeze().numpy()
        colored_mask = np.zeros((*mask_np.shape, 4))
        color = plt.cm.tab10(idx % 10)[:3]
        colored_mask[mask_np > 0.5] = [*color, 0.5]
        axes[1].imshow(colored_mask)
    axes[1].set_title("Segmentation Masks")
    axes[1].axis('off')

    axes[2].imshow(image_np)
    for idx, (box, score) in enumerate(zip(boxes[:5], scores[:5])):  # Limiting to first 5 for clarity
        x1, y1, x2, y2 = box.numpy()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, edgecolor=plt.cm.tab10(idx), linewidth=2)
        axes[2].add_patch(rect)
        axes[2].text(x1, y1 - 5, f'{score.item():.2f}',
                     color='white', fontsize=10,
                     bbox=dict(facecolor=plt.cm.tab10(idx), alpha=0.8))
    axes[2].set_title("Bounding Boxes with Scores")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig) # Close the figure to free memory
    print(f"SAM3 results saved to {save_path}")

draw_and_save_sam3_results(image, masks, boxes, scores, 'sam3_results.png')


#################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]