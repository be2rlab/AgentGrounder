import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Server/headless

def draw_and_save_sam3_results(image, masks, boxes, scores, prompt, save_path='sam3_results.png'):
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
    # axes[0].set_title("Original Image")
    axes[0].set_title(prompt)
    axes[0].axis('off')

    axes[1].imshow(image_np)
    for idx, mask in enumerate(masks[:5]): # Limiting to first 5 for clarity
        mask_np = mask.squeeze().numpy()
        colored_mask = np.zeros((*mask_np.shape, 4))
        color = plt.cm.tab10(idx % 10)[:3]
        colored_mask[mask_np > 0.5] = [*color, 0.5]
        axes[1].imshow(colored_mask)
    # axes[1].set_title(f"Segmentation Masks: {prompt}")
    axes[1].axis('off')

    axes[2].imshow(image_np)
    for idx, (box, score) in enumerate(zip(boxes[:5], scores[:5])): # Limiting to first 5 for clarity
        x1, y1, x2, y2 = box.numpy()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=plt.cm.tab10(idx), linewidth=2)
        axes[2].add_patch(rect)
        axes[2].text(x1, y1 - 5, f'{score.item():.2f}', color='white', fontsize=10, bbox=dict(facecolor=plt.cm.tab10(idx), alpha=0.8))
    # axes[2].set_title(f"Bounding Boxes with Scores: {prompt}")
    axes[2].axis('off')

    # plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig) # Close the figure to free memory
    print(f"SAM3 results saved to {save_path}")
    

def extract_and_project(pcd, bbox, output_path="projections.png"):
    """
    pcd: np.array of shape (n, 6) -> x, y, z, r, g, b
    bbox: tuple (cx, cy, cz, h, w, d)
    """
    cx, cy, cz, h, w, d = bbox
    
    # 1. Define boundaries
    x_min, x_max = cx - w/2, cx + w/2
    y_min, y_max = cy - h/2, cy + h/2
    z_min, z_max = cz - d/2, cz + d/2
    
    # 2. Filter points (Crop the object)
    mask = (
        (pcd[:, 0] >= x_min) & (pcd[:, 0] <= x_max) &
        (pcd[:, 1] >= y_min) & (pcd[:, 1] <= y_max) &
        (pcd[:, 2] >= z_min) & (pcd[:, 2] <= z_max)
    )
    obj_pcd = pcd[mask]
    
    if len(obj_pcd) == 0:
        print("No points found inside the bounding box.")
        return None

    # 3. Setup projections
    # Views: Front, Back, Left, Right, Top, Bottom
    projections = [
        (0, 1, "Front (XY)"), (0, 1, "Back (XY)"),
        (1, 2, "Left (YZ)"),  (1, 2, "Right (YZ)"),
        (0, 2, "Top (XZ)"),   (0, 2, "Bottom (XZ)")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (dim1, dim2, title) in enumerate(projections):
        axes[i].scatter(obj_pcd[:, dim1], obj_pcd[:, dim2], s=1, c=obj_pcd[:, 3:6] if obj_pcd.shape[1] >= 6 else 'blue')
        axes[i].set_title(title)
        axes[i].set_aspect('equal')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Projections saved to {output_path}")
    
    return obj_pcd


def extract_and_project_heavy(pcd, bbox, output_path="projections.png"):
    """
    6-view 3D scene with THICK RED 3D bbox lines + points. SERVER SAFE.
    """
    
    # Bbox: (cx,cy,cz,h,w,d)
    cx, cy, cz, hgt, wid, dep = bbox
    min_bound = np.array([cx-wid/2, cy-hgt/2, cz-dep/2])
    max_bound = np.array([cx+wid/2, cy+hgt/2, cz+dep/2])
    
    # 8 corners of bbox
    corners = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],  # 0
        [max_bound[0], min_bound[1], min_bound[2]],  # 1
        [max_bound[0], max_bound[1], min_bound[2]],  # 2  
        [min_bound[0], max_bound[1], min_bound[2]],  # 3
        [min_bound[0], min_bound[1], max_bound[2]],  # 4
        [max_bound[0], min_bound[1], max_bound[2]],  # 5
        [max_bound[0], max_bound[1], max_bound[2]],  # 6
        [min_bound[0], max_bound[1], max_bound[2]]   # 7
    ])
    
    # 12 EDGES - thick red 3D lines
    edges = [
        [0,1], [1,2], [2,3], [3,0],  # bottom face
        [4,5], [5,6], [6,7], [7,4],  # top face
        [0,4], [1,5], [2,6], [3,7]   # verticals
    ]
    
    points = pcd[:, :3]
    colors = pcd[:, 3:6]
    
    fig = plt.figure(figsize=(18, 12))
    axes = [fig.add_subplot(2, 3, i+1, projection='3d') for i in range(6)]
    fig.suptitle('3D Point Cloud + RED Bounding Box (Server Render)', fontsize=18)
    
    # 6 views with perfect 3D bbox visibility
    views = [
        ('Front',  20,  -90), ('Back',   20,   90),
        ('Left',   20,    0), ('Right',  20,  180),
        ('Top',    90,    0), ('Bottom',-90,    0)
    ]
    
    for i, (title, elev, azim) in enumerate(views):
        ax = axes[i]
        
        # Plot COLORED POINTS
        ax.scatter(points[:,0], points[:,1], points[:,2], 
                  c=colors, s=30, alpha=0.8)
        
        # Plot THICK RED 3D LINES for bbox
        for edge in edges:
            pts = corners[edge]
            ax.plot(pts[:,0], pts[:,1], pts[:,2], 
                   'r-', linewidth=8, alpha=0.9)
        
        # Camera position
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Clean view - focus on bbox
        ax.set_xlim(min_bound[0], max_bound[0])
        ax.set_ylim(min_bound[1], max_bound[1])
        ax.set_zlim(min_bound[2], max_bound[2])
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 6-view 3D render with THICK RED BBOX saved: {output_path}")
