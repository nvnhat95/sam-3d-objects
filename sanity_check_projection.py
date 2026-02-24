import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# import inference code
import sys
sys.path.append("notebook")
from inference import Inference, load_image

# 3D projection utilities
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from sam3d_objects.pipeline.layout_post_optimization_utils import (
    get_mesh,
    get_mask_renderer,
    compute_iou,
)

def load_mask(pt_path, image_name):
    # Load the big pt dictionary
    mask_dict = torch.load(pt_path)
    
    # Try different keys
    basename = os.path.basename(image_name)
    basename_no_ext = os.path.splitext(basename)[0]
    
    if basename in mask_dict:
        mask = mask_dict[basename]
    elif basename_no_ext in mask_dict:
        mask = mask_dict[basename_no_ext]
    elif image_name in mask_dict:
        mask = mask_dict[image_name]
    else:
        # Debugging: show available keys
        keys = list(mask_dict.keys())
        raise ValueError(f"Could not find mask for {image_name}. Available keys example: {keys[:5]}")
        
    return mask

def get_camera_intrinsics_from_blender(transform_frame, width, height):
    # Depending on how it's stored, typically standard intrinsics
    # Assuming camera_angle_x is available globally, but let's try to extract if present
    # Usually in NeRF/Blender datasets, it's defined per frame or globally
    # If not present in frame, we might need it from global json. We'll attempt a common fallback.
    pass

def main():
    parser = argparse.ArgumentParser(description="Test SAM3D Projection")
    parser.add_argument("--config_path", type=str, required=True, help="Path to SAM3D config yaml")
    parser.add_argument("--blender_dir", type=str, required=True, help="Path to blender dataset directory")
    parser.add_argument("--mask_pt", type=str, required=True, help="Path to mask .pt file")
    args = parser.parse_args()

    # 1. Load transforms_train.json
    transforms_path = os.path.join(args.blender_dir, "transforms_train.json")
    with open(transforms_path, "r") as f:
        transforms = json.load(f)
        
    # Get the first frame
    frame = transforms["frames"][0]
    # File path usually looks like "./train/r_0"
    file_path = frame["file_path"]
    if not file_path.endswith(".png"):
        file_path += ".png"
    
    # Remove leading ./
    if file_path.startswith("./"):
        file_path = file_path[2:]
        
    full_image_path = os.path.join(args.blender_dir, file_path)
    print(f"Loading image from {full_image_path}")
    
    # 2. Load Image
    image = load_image(full_image_path)
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.uint8)
    h, w = image.shape[:2]
    print(f"Image shape: {image.shape}")
    
    # Extract intrinsics
    camera_angle_x = transforms.get("camera_angle_x", 0.6911112070083618)
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    
    # Create normalized intrinsics (expected by get_mask_renderer which calls denormalize_f)
    intrinsics = torch.tensor([
        [focal / w, 0, 0.5],
        [0, focal / h, 0.5],
        [0, 0, 1]
    ], dtype=torch.float32)
    print(f"Intrinsics (normalized):\n{intrinsics}")
    
    # 3. Load Mask
    mask_np = load_mask(args.mask_pt, file_path)
    mask = mask_np.astype(bool)
    print(f"Loaded mask, sum: {mask.sum()}, shape: {mask.shape}")
    
    # 4. SAM3D Inference
    print("Initializing SAM3D Inference pipeline...")
    inference = Inference(args.config_path, compile=False)
    
    print("Running Inference...")
    # Make sure to run with with_layout_postprocess=False because we want the raw output to project
    output = inference(image, mask, seed=42)
    
    glb = output["glb"]
    R_quat = output["rotation"]
    S = output["scale"]
    T = output["translation"]
    
    device = R_quat.device
    
    print(f"Outputs extracted: glb Mesh, R: {R_quat.shape}, S: {S.shape}, T: {T.shape}")
    
    # 5. Projection back to 2D
    print("Projecting 3D Mesh back to 2D image plane...")
    # Convert quaternion to rotation matrix
    Rotation = quaternion_to_matrix(R_quat.squeeze(1))
    
    # Compose transformation
    tfm_ori = compose_transform(scale=S, rotation=Rotation, translation=T)
    
    # Convert trimesh/o3d (glb) to PyTorch3D meshes, applying the transformation
    mesh, faces_idx, textures = get_mesh(glb, tfm_ori, device)
    
    # Convert boolean mask to float tensor for getting renderer
    mask_tensor = torch.from_numpy(mask).float().to(device)
    
    # Get mask and renderer
    H, W = mask.shape
    mask_resized, renderer = get_mask_renderer(mask_tensor, min(H, W), intrinsics, device)
    
    # Render
    with torch.no_grad():
        rendered = renderer(mesh)
        
    pred_mask = rendered[..., 3][0] # Silhouette is in the 4th channel (Alpha), batch 0
    
    # 6. Compute IoU
    iou = compute_iou(pred_mask[None, None], mask_resized, threshold=0.5)
    print(f"IoU: {iou.item():.4f}")
    
    # 7. Plot results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Mask")
    plt.imshow(mask_resized[0, 0].cpu().numpy(), cmap='gray')
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Projected Mask (IoU: {iou.item():.4f})")
    plt.imshow(pred_mask.cpu().numpy(), cmap='gray')
    plt.axis("off")
    
    plot_path = "projection_plot.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
