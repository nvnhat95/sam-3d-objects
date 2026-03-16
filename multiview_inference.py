"""
Multi-view 3D Inference
=======================
Runs SAM-3D on N camera views from a Blender NeRF dataset, fuses the
per-view sparse structures in world space (majority vote), then decodes
one 3D object per view conditioned on the averaged structure and renders
an N×N projection grid.

Usage:
    python multi_view_inference.py \
        --config_path /path/to/config.yaml \
        --blender_dir  /path/to/blender_dataset \
        --mask_pt      /path/to/masks.pt \
        --n_views      4 \
        --seed         42 \
        --output_dir   multiview_output
"""

import argparse
import copy
import json
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix

sys.path.append("notebook")
from inference import Inference, load_image  # noqa: E402

from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from sam3d_objects.pipeline.layout_post_optimization_utils import (
    compute_iou,
    get_mask_renderer,
    get_mesh,
)

# ---------------------------------------------------------------------------
# Coordinate-system constants
# ---------------------------------------------------------------------------
# Model camera space (PyTorch3D / R3): x-right, y-down, z-forward
# Blender camera space (OpenGL):        x-right, y-up,   z-backward
# Conversion between them: flip Y and Z.
_FLIP_YZ = torch.tensor([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_mask_from_pt(pt_path: str, image_name: str) -> np.ndarray:
    """Look up a boolean mask for *image_name* inside a .pt dictionary."""
    mask_dict = torch.load(pt_path)
    basename = os.path.basename(image_name)
    basename_no_ext = os.path.splitext(basename)[0]
    for key in (basename, basename_no_ext, image_name):
        if key in mask_dict:
            return np.asarray(mask_dict[key], dtype=bool)
    keys_sample = list(mask_dict.keys())[:5]
    raise ValueError(
        f"Cannot find mask for '{image_name}'. "
        f"Available key examples: {keys_sample}"
    )


def build_intrinsics(camera_angle_x: float, w: int, h: int) -> torch.Tensor:
    """Normalized 3×3 intrinsic matrix expected by *get_mask_renderer*."""
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    return torch.tensor(
        [[focal / w, 0.0, 0.5], [0.0, focal / h, 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def voxels_local_to_world(
    coords: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    R_c2w: torch.Tensor,
    T_c2w: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Transform voxel indices from object-local space to world space.

    Local space convention (same as model output):
        coords[:, 1:]  in [0, 63]  →  v_local = coords/64 − 0.5  ∈ [−0.5, 0.5]

    Chain:  local → PyTorch3D cam  (via predicted pose S/R/T)
                 → Blender cam     (flip Y and Z)
                 → world           (via blender cam-to-world matrix)

    Args:
        coords:      (M, 4) int – [batch, x, y, z] indices in [0, 63]
        rotation:    (1, 1, 4) wxyz quaternion from stage-1 inference
        translation: (1, 3)
        scale:       (1, 3)
        R_c2w:       (3, 3) Blender camera-to-world rotation
        T_c2w:       (3,)   Blender camera-to-world translation
    Returns:
        (M, 3) float32 – world-space voxel positions
    """
    flip = _FLIP_YZ.to(device)
    v_local = (coords[:, 1:].float() / 64.0 - 0.5).to(device)          # (M, 3)

    R_mat = quaternion_to_matrix(rotation.squeeze(1))                    # (1, 3, 3)
    tfm = compose_transform(scale, R_mat, translation)
    v_p3d = tfm.transform_points(v_local.unsqueeze(0))[0]               # (M, 3)

    v_blender = v_p3d * flip                                             # P3D → Blender cam
    # Row-vector: p_world = p_cam @ R_c2w.T + T_c2w
    v_world = v_blender @ R_c2w.T.to(device) + T_c2w.to(device)        # (M, 3)
    return v_world


def world_to_local_coords(
    world_pts: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    R_c2w: torch.Tensor,
    T_c2w: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Map world-space points back to integer voxel indices [0, 63].

    Inverse of *voxels_local_to_world*.

    Returns:
        (K', 4) int32 – unique [batch=0, x, y, z] coords in [0, 63]
    """
    flip = _FLIP_YZ.to(device)
    world_pts = world_pts.to(device)
    R_c2w = R_c2w.to(device)
    T_c2w = T_c2w.to(device)

    # World → Blender cam: p_cam = (p_world − T) @ R_c2w
    v_blender = (world_pts - T_c2w) @ R_c2w                             # (K, 3)
    v_p3d = v_blender * flip                                             # Blender → P3D cam

    R_mat = quaternion_to_matrix(rotation.squeeze(1))                    # (1, 3, 3)
    tfm = compose_transform(scale, R_mat, translation)
    v_local = tfm.inverse().transform_points(v_p3d.unsqueeze(0))[0]     # (K, 3)

    v_idx = ((v_local + 0.5) * 64.0).long().clamp(0, 63)               # (K, 3)
    batch_col = torch.zeros(v_idx.shape[0], 1, dtype=torch.int32, device=device)
    coords = torch.cat([batch_col, v_idx.int()], dim=1)                  # (K, 4)
    return coords.unique(dim=0)


# ---------------------------------------------------------------------------
# Sparse-structure fusion
# ---------------------------------------------------------------------------

def average_sparse_structures(
    all_world_pts: list,
    resolution: int = 32,
    min_votes: Optional[int] = None,
):
    """Fuse per-view world-space voxel clouds via majority vote.

    Args:
        all_world_pts: N tensors of shape (M_i, 3) – world-space voxel centres
        resolution:    side length of the world-space voting grid
        min_votes:     minimum number of views a cell must receive to be kept
                       (defaults to ceil(N / 2))
    Returns:
        voxel_centres_world:  (K, 3) – kept world-space voxel centres
        bbox_min:             (3,)
        voxel_size:           (3,)
    """
    N = len(all_world_pts)
    if min_votes is None:
        min_votes = max(1, (N + 1) // 2)

    all_pts = torch.cat(all_world_pts, dim=0)                           # (total, 3)
    bbox_min = all_pts.min(dim=0).values
    bbox_max = all_pts.max(dim=0).values

    # Small padding to avoid boundary issues
    pad = (bbox_max - bbox_min) * 0.05 + 1e-6
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad
    voxel_size = (bbox_max - bbox_min) / resolution                     # (3,)

    vote_grid = torch.zeros(resolution ** 3, dtype=torch.int32)

    for pts in all_world_pts:
        pts = pts.cpu()
        idx = ((pts - bbox_min) / voxel_size).long().clamp(0, resolution - 1)
        flat = idx[:, 0] * resolution * resolution + idx[:, 1] * resolution + idx[:, 2]
        flat_unique = flat.unique()
        # One vote per unique cell per view
        vote_grid.scatter_add_(
            0,
            flat_unique,
            torch.ones(flat_unique.shape[0], dtype=torch.int32),
        )

    vote_grid = vote_grid.view(resolution, resolution, resolution)
    occ_idx = (vote_grid >= min_votes).nonzero(as_tuple=False)          # (K, 3)

    if occ_idx.shape[0] == 0:
        # Fall back to union if majority vote yields nothing
        print("  [warn] majority vote produced 0 voxels – falling back to union")
        occ_idx = (vote_grid >= 1).nonzero(as_tuple=False)

    voxel_centres = (occ_idx.float() + 0.5) * voxel_size + bbox_min    # (K, 3)
    return voxel_centres, bbox_min, voxel_size


# ---------------------------------------------------------------------------
# Cross-view mesh projection
# ---------------------------------------------------------------------------

def transform_verts_cam_i_to_cam_j(
    verts_cam_i: torch.Tensor,
    R_c2w_i: torch.Tensor,
    T_c2w_i: torch.Tensor,
    R_c2w_j: torch.Tensor,
    T_c2w_j: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Re-express mesh vertices from PyTorch3D-cam-i space into PyTorch3D-cam-j.

    Chain:  P3D cam_i → Blender cam_i → world → Blender cam_j → P3D cam_j
    """
    flip = _FLIP_YZ.to(device)
    R_c2w_i = R_c2w_i.to(device)
    T_c2w_i = T_c2w_i.to(device)
    R_c2w_j = R_c2w_j.to(device)
    T_c2w_j = T_c2w_j.to(device)

    v_blender_i = verts_cam_i * flip                                     # P3D → Blender cam_i
    v_world = v_blender_i @ R_c2w_i.T + T_c2w_i                        # → world
    v_blender_j = (v_world - T_c2w_j) @ R_c2w_j                        # → Blender cam_j
    v_p3d_j = v_blender_j * flip                                         # → P3D cam_j
    return v_p3d_j


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SAM-3D multi-view inference")
    parser.add_argument("--config_path", required=True, help="Path to SAM3D config YAML")
    parser.add_argument("--blender_dir", required=True, help="Blender NeRF dataset directory")
    parser.add_argument("--mask_pt", required=True, help="Path to masks .pt file")
    parser.add_argument("--n_views", type=int, default=4, help="Number of camera views")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="multiview_output")
    parser.add_argument(
        "--vote_resolution",
        type=int,
        default=32,
        help="Side length of the world-space voting grid",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load frames ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading frames from blender dataset")
    transforms_path = os.path.join(args.blender_dir, "transforms_train.json")
    with open(transforms_path) as f:
        transforms = json.load(f)

    all_frames = transforms["frames"]
    total = len(all_frames)
    indices = np.linspace(0, total - 1, args.n_views, dtype=int).tolist()
    selected_frames = [all_frames[i] for i in indices]
    camera_angle_x = transforms.get("camera_angle_x", 0.6911112070083618)

    frame_data = []
    for frame in selected_frames:
        file_path = frame["file_path"]
        if not file_path.endswith(".png"):
            file_path += ".png"
        if file_path.startswith("./"):
            file_path = file_path[2:]

        full_path = os.path.join(args.blender_dir, file_path)
        image = load_image(full_path)                                    # (H, W, 3|4) uint8
        h, w = image.shape[:2]

        mask_np = load_mask_from_pt(args.mask_pt, file_path)            # (H, W) bool

        intrinsics = build_intrinsics(camera_angle_x, w, h)

        transform_matrix = np.array(frame["transform_matrix"])
        R_c2w = torch.from_numpy(transform_matrix[:3, :3]).float()
        T_c2w = torch.from_numpy(transform_matrix[:3, 3]).float()

        # Merge mask into the alpha channel → RGBA uint8
        mask_uint8 = mask_np.astype(np.uint8) * 255
        rgba = np.concatenate([image[..., :3], mask_uint8[..., None]], axis=-1)

        frame_data.append(
            dict(
                file_path=file_path,
                image=image,
                rgba=rgba,
                mask=mask_np,
                intrinsics=intrinsics,
                R_c2w=R_c2w,
                T_c2w=T_c2w,
            )
        )
        print(f"  Loaded {file_path}  shape={image.shape}  mask_area={mask_np.sum()}")

    # ── 2. Load pipeline ────────────────────────────────────────────────────
    print("\nStep 2: Loading SAM-3D pipeline")
    infer = Inference(args.config_path, compile=False)
    pipeline = infer._pipeline
    device = pipeline.device
    print(f"  Device: {device}")

    # ── 3. Stage 1: per-view sparse-structure inference ─────────────────────
    print("\nStep 3: Stage-1 (sparse structure) inference")
    stage1_results = []
    for i, fd in enumerate(frame_data):
        print(f"  View {i}: {fd['file_path']}")
        torch.manual_seed(args.seed)
        result = pipeline.run(
            fd["rgba"],
            None,
            seed=args.seed,
            stage1_only=True,
        )
        stage1_results.append(
            dict(
                coords=result["coords"],           # (M, 4) [batch, x, y, z]
                rotation=result["rotation"],       # (1, 1, 4) wxyz
                translation=result["translation"], # (1, 3)
                scale=result["scale"],             # (1, 3)
            )
        )
        print(
            f"    coords: {result['coords'].shape[0]} voxels  "
            f"scale={result['scale'].tolist()}"
        )

    # ── 4. Transform voxels to world space ──────────────────────────────────
    print("\nStep 4: Transforming voxels to world space")
    all_world_pts = []
    for i, (fd, r1) in enumerate(zip(frame_data, stage1_results)):
        wpts = voxels_local_to_world(
            r1["coords"],
            r1["rotation"],
            r1["translation"],
            r1["scale"],
            fd["R_c2w"],
            fd["T_c2w"],
            device,
        )
        all_world_pts.append(wpts.cpu())
        print(
            f"  View {i}: {wpts.shape[0]} world-space voxels  "
            f"bbox [{wpts.min(0).values.tolist()} … {wpts.max(0).values.tolist()}]"
        )

    # ── 5. Average sparse structure (majority vote) ─────────────────────────
    print(f"\nStep 5: Majority-vote fusion (resolution={args.vote_resolution})")
    min_votes = max(1, (args.n_views + 1) // 2)
    voxel_centres_world, bbox_min, voxel_size = average_sparse_structures(
        all_world_pts,
        resolution=args.vote_resolution,
        min_votes=min_votes,
    )
    print(
        f"  Averaged structure: {voxel_centres_world.shape[0]} voxels "
        f"(min_votes={min_votes})"
    )

    # ── 6. Stage 2: SLAT decode with averaged coords ─────────────────────────
    print("\nStep 6: Stage-2 (SLAT decode) with averaged sparse structure")
    decoded_outputs = []
    for i, (fd, r1) in enumerate(zip(frame_data, stage1_results)):
        print(f"  View {i}: {fd['file_path']}")

        # Map world-average voxels back to this view's local coordinate space
        avg_coords_i = world_to_local_coords(
            voxel_centres_world,
            r1["rotation"],
            r1["translation"],
            r1["scale"],
            fd["R_c2w"],
            fd["T_c2w"],
            device,
        )
        print(f"    Averaged coords → {avg_coords_i.shape[0]} voxels for view {i}")

        if avg_coords_i.shape[0] == 0:
            print("    [warn] No averaged coords; falling back to stage-1 coords")
            avg_coords_i = r1["coords"]

        # Preprocess the image for the SLAT stage (no pointmap needed)
        slat_input = pipeline.preprocess_image(fd["rgba"], pipeline.slat_preprocessor)

        torch.manual_seed(args.seed)
        slat = pipeline.sample_slat(slat_input, avg_coords_i)
        decoded = pipeline.decode_slat(slat, formats=["mesh", "gaussian"])
        decoded = pipeline.postprocess_slat_output(
            decoded,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
        )

        decoded_outputs.append(
            dict(
                glb=decoded.get("glb"),
                gaussian=decoded.get("gs"),
                rotation=r1["rotation"],
                translation=r1["translation"],
                scale=r1["scale"],
            )
        )
        print(f"    Decoded: glb={'yes' if decoded.get('glb') else 'no'}  "
              f"gs={'yes' if decoded.get('gs') else 'no'}")

    # ── 7. N×N visualization ─────────────────────────────────────────────────
    N = args.n_views
    print(f"\nStep 7: Rendering {N}×{N} projection grid")

    # Build one renderer per camera view
    renderers = []
    for fd in frame_data:
        mask_t = torch.from_numpy(fd["mask"]).float().to(device)
        H, W = fd["mask"].shape
        _, renderer = get_mask_renderer(mask_t, min(H, W), fd["intrinsics"], device)
        renderers.append(renderer)

    fig, axes = plt.subplots(N, N, figsize=(4 * N, 4 * N))
    # Normalise axes to a list-of-lists regardless of N
    if N == 1:
        axes = [[axes]]
    else:
        axes = [list(row) for row in axes]

    iou_table = np.full((N, N), float("nan"))

    for i, out_i in enumerate(decoded_outputs):
        glb_i = out_i["glb"]
        if glb_i is None:
            print(f"  Object {i}: no GLB mesh – skipping row")
            continue

        R_i = quaternion_to_matrix(out_i["rotation"].squeeze(1))        # (1, 3, 3)
        tfm_obj2cam_i = compose_transform(
            out_i["scale"], R_i, out_i["translation"]
        )

        # Build the mesh in cam_i PyTorch3D space once.
        # get_mesh also applies the z-up → y-up rotation to GLB vertices.
        mesh_cam_i, _, _ = get_mesh(copy.deepcopy(glb_i), tfm_obj2cam_i, device)
        verts_cam_i = mesh_cam_i.verts_list()[0]                        # (V, 3)
        faces_i = mesh_cam_i.faces_list()[0]                            # (F, 3)
        fd_i = frame_data[i]

        for j, fd_j in enumerate(frame_data):
            print(f"  Object {i} → Camera {j} …", end=" ", flush=True)

            if i == j:
                verts_cam_j = verts_cam_i
            else:
                verts_cam_j = transform_verts_cam_i_to_cam_j(
                    verts_cam_i,
                    fd_i["R_c2w"], fd_i["T_c2w"],
                    fd_j["R_c2w"], fd_j["T_c2w"],
                    device,
                )

            # Render silhouette in camera j
            textures_j = TexturesVertex(
                verts_features=torch.ones_like(verts_cam_j)[None]
            )
            mesh_cam_j = Meshes(
                verts=[verts_cam_j],
                faces=[faces_i],
                textures=textures_j,
            )

            with torch.no_grad():
                rendered = renderers[j](mesh_cam_j)

            # Silhouette is in the alpha channel (index 3)
            pred_mask = rendered[..., 3][0].cpu()                       # (H, W)

            # Compute IoU against view j's GT mask (resized to renderer output size)
            mask_t_j = torch.from_numpy(fd_j["mask"]).float()
            mask_for_iou = torch.nn.functional.interpolate(
                mask_t_j[None, None].to(device),
                size=pred_mask.shape,
                mode="bilinear",
                align_corners=False,
            )
            iou_val = compute_iou(
                pred_mask[None, None].to(device), mask_for_iou, threshold=0.5
            )
            iou_table[i, j] = iou_val.item()
            print(f"IoU={iou_val.item():.3f}")

            # Save individual image
            save_path = os.path.join(args.output_dir, f"view{i}_proj{j}.png")
            plt.imsave(save_path, pred_mask.numpy(), cmap="gray")

            # Write to grid
            ax = axes[i][j]
            ax.imshow(pred_mask.numpy(), cmap="gray")
            ax.set_title(f"Obj {i} → Cam {j}\nIoU={iou_val.item():.3f}", fontsize=8)
            ax.axis("off")

    fig.suptitle(
        f"Multi-view projection  ({N} views)\n"
        f"Rows: decoded object index  |  Cols: camera view index",
        fontsize=11,
    )
    fig.tight_layout()
    grid_path = os.path.join(args.output_dir, "multiview_grid.png")
    fig.savefig(grid_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"\nSaved N×N grid  →  {grid_path}")

    # Print IoU table
    print("\nIoU table (row = object index, col = camera index):")
    header = "       " + "".join(f"  Cam{j}" for j in range(N))
    print(header)
    for i in range(N):
        row = f"  Obj{i} " + "".join(f" {iou_table[i, j]:5.3f}" for j in range(N))
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
