"""
Multi-view Consistent 3D Shape Optimization via SAM3D Architecture.

Optimizes shared layout and shape tokens for multi-view consistency.
Given a Blender scene with multiple camera views and object masks,
this script finds a 3D voxel shape and per-view poses such that
projecting the shape to each view matches the corresponding mask.

Architecture (Geometry Model — see diagram):
    Image + Mask → Image Encoder (DINOv2, frozen) → Image Tokens
    Layout Tokens (learnable, shared) ─┬→ Mixture of Transformers* ─→ Layout Decoder → R, T, S
    Shape Tokens  (learnable, shared) ─┘                            → Shape Decoder  → Voxel (64³)

Training loop:
    1. Select 2 camera views (i, j)
    2. Forward view_i through model → voxel shape + layout_i (R, T, S)
    3. Project voxel to view_j using known camera extrinsics → rendered_mask_j
    4. Compute IoU loss between rendered_mask_j and GT mask_j
    5. Backpropagate to update layout tokens and shape logits

Usage (from-scratch):
    python sam_3d_consistent.py \\
        --blender_dir /path/to/blender/scene \\
        --mask_pt /path/to/masks.pt \\
        --num_steps 5000

Usage (finetuning with SAM3D checkpoint):
    python sam_3d_consistent.py \\
        --config_path checkpoints/hf/pipeline.yaml \\
        --blender_dir /path/to/blender/scene \\
        --mask_pt /path/to/masks.pt \\
        --num_steps 5000 \\
        --freeze_backbone
"""

import os
import json
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# ============================================================================
#  Monkey Patch for deepcopy issue in sam3d_objects
# ============================================================================

try:
    import sam3d_objects.data.dataset.tdfy.pose_target as pose_target_module
    
    def _patched_dicts_pose_target_to_instance_pose(**kwargs):
        pose_target_convention = kwargs.get("pose_target_convention")
        # In original code this was globals()[pose_target_convention]
        _convention_class = getattr(pose_target_module, pose_target_convention)
        assert (
            _convention_class.pose_target_convention == pose_target_convention
        ), f"Normalization name mismatch: {_convention_class.pose_target_convention} != {pose_target_convention}"

        normalize = kwargs.pop("normalize", False)
        pose_target = pose_target_module.PoseTarget(**kwargs)
        instance_pose = pose_target_module.PoseTargetConverter.pose_target_to_instance_pose(pose_target, normalize)
        
        # Fix: avoid asdict() which does deepcopy on tensors (causing RuntimeError)
        return dict(instance_pose.__dict__)

    pose_target_module.PoseTargetConverter.dicts_pose_target_to_instance_pose = staticmethod(_patched_dicts_pose_target_to_instance_pose)
    print("[MonkeyPatch] Applied fix for PoseTargetConverter deepcopy error.")
except ImportError:
    # Package might not be installed or path issues; skip silently or print warning
    print("[MonkeyPatch] Could not import sam3d_objects.data.dataset.tdfy.pose_target, skipping patch.")
except Exception as e:
    print(f"[MonkeyPatch] Failed to apply fix: {e}")


# ============================================================================
#  Utility Functions
# ============================================================================

def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Reference: Zhou et al., "On the Continuity of Rotation Representations
    in Neural Networks", CVPR 2019.

    Args:
        rot_6d: (..., 6) tensor — two 3D vectors packed together.
    Returns:
        (..., 3, 3) proper rotation matrix.
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3) columns = b1, b2, b3


def build_layout_matrix(
    R: torch.Tensor, T: torch.Tensor, S: torch.Tensor
) -> torch.Tensor:
    """Build 4x4 affine matrix: p_cam = R @ diag(S) @ p_obj + T.

    Args:
        R: (B, 3, 3) rotation matrix.
        T: (B, 3) translation.
        S: (B, 3) positive scale.
    Returns:
        (B, 4, 4) affine transformation matrix.
    """
    B = R.shape[0]
    device = R.device
    # RS[:, :, j] = R[:, :, j] * S[:, j]  →  R @ diag(S)
    RS = R * S[:, None, :]  # (B, 3, 3)
    M = torch.eye(4, device=device, dtype=R.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    M[:, :3, :3] = RS
    M[:, :3, 3] = T
    return M


def blender_c2w_to_opencv(c2w: torch.Tensor) -> torch.Tensor:
    """Convert Blender/OpenGL c2w to OpenCV camera convention.

    Blender (OpenGL): X-right, Y-up, Z-towards-viewer.
    OpenCV:           X-right, Y-down, Z-forward.
    We flip the Y and Z axes of the camera coordinate system.
    """
    flip = torch.tensor(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=c2w.dtype,
        device=c2w.device,
    )
    return c2w @ flip


def soft_iou(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Differentiable IoU for soft [0, 1] predicted masks and binary GT masks.

    Args:
        pred:   (B, 1, H, W) soft mask in [0, 1].
        target: (B, 1, H, W) binary ground-truth mask.
    Returns:
        Scalar IoU averaged over batch.
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=-1)
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1) - intersection
    iou = intersection / (union + eps)
    return iou.mean()


# ============================================================================
#  Dataset
# ============================================================================

class BlenderMaskDataset(Dataset):
    """Blender-rendered multi-view dataset with per-image object masks.

    Expected Blender directory layout::

        blender_dir/
        ├── transforms_train.json   (or transforms.json)
        └── <images referenced by file_path in JSON>

    Mask file: a ``.pt`` dict  ``{image_name_str: np.ndarray(bool)}``.
    """

    def __init__(self, blender_dir: str, mask_pt_path: str, image_size: int = 256):
        self.blender_dir = Path(blender_dir)
        self.image_size = image_size

        # --- Load transforms JSON ---
        transforms_path = self.blender_dir / "transforms_train.json"
        if not transforms_path.exists():
            transforms_path = self.blender_dir / "transforms.json"
        if not transforms_path.exists():
            raise FileNotFoundError(f"No transforms JSON found in {blender_dir}")
        with open(transforms_path) as f:
            self.meta = json.load(f)

        # --- Intrinsics ---
        self.camera_angle_x = self.meta.get("camera_angle_x", None)
        self.fl_x = self.meta.get("fl_x", None)
        self.fl_y = self.meta.get("fl_y", None)
        self.cx = self.meta.get("cx", None)
        self.cy = self.meta.get("cy", None)

        # --- Load masks ---
        mask_data = torch.load(mask_pt_path, weights_only=False)
        self.masks: dict[str, torch.Tensor] = {}
        for k, v in mask_data.items():
            if isinstance(v, np.ndarray):
                self.masks[k] = torch.from_numpy(v.astype(np.float32))
            elif isinstance(v, torch.Tensor):
                self.masks[k] = v.float()
            else:
                raise TypeError(f"Unsupported mask type {type(v)} for key '{k}'")

        # --- Filter frames that have a matching mask ---
        self.frames: list[dict] = []
        for frame in self.meta["frames"]:
            name = self._get_frame_name(frame)
            if self._find_mask(name) is not None:
                self.frames.append(frame)
        print(
            f"[Dataset] Loaded {len(self.frames)} frames with masks "
            f"(out of {len(self.meta['frames'])} total)"
        )

    # --- helpers ---

    @staticmethod
    def _get_frame_name(frame: dict) -> str:
        return Path(frame["file_path"]).stem

    def _find_mask(self, name: str):
        """Lookup mask by trying several common key formats."""
        for key in [name, f"{name}.png", f"{name}.jpg", f"{name}.jpeg"]:
            if key in self.masks:
                return self.masks[key]
        return None

    def get_intrinsics(self, H: int | None = None, W: int | None = None) -> torch.Tensor:
        """Return (3, 3) camera intrinsic matrix."""
        H = H or self.image_size
        W = W or self.image_size
        if self.fl_x is not None:
            fx, fy = self.fl_x, (self.fl_y or self.fl_x)
            cx_val = self.cx if self.cx is not None else W / 2
            cy_val = self.cy if self.cy is not None else H / 2
        elif self.camera_angle_x is not None:
            fx = 0.5 * W / math.tan(0.5 * self.camera_angle_x)
            fy = fx
            cx_val, cy_val = W / 2, H / 2
        else:
            raise ValueError("No intrinsics information in transforms JSON")
        return torch.tensor(
            [[fx, 0, cx_val], [0, fy, cy_val], [0, 0, 1]], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        frame = self.frames[idx]
        name = self._get_frame_name(frame)

        # --- Image ---
        img_path = self.blender_dir / frame["file_path"]
        if not img_path.suffix:
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = img_path.with_suffix(ext)
                if candidate.exists():
                    img_path = candidate
                    break
        image = Image.open(str(img_path)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0

        # --- Mask ---
        mask = self._find_mask(name)
        assert mask is not None, f"Mask not found for frame '{name}'"
        if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
            mask = F.interpolate(
                mask[None, None],
                size=(self.image_size, self.image_size),
                mode="nearest",
            )[0, 0]

        # --- Camera (convert Blender/OpenGL → OpenCV) ---
        c2w_gl = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w_cv = blender_c2w_to_opencv(c2w_gl)

        return {"image": image, "mask": mask, "c2w": c2w_cv, "name": name}


# ============================================================================
#  SAM3D checkpoint loading and finetuning wrapper
# ============================================================================

def load_sam3d_pipeline(config_path: str, compile_model: bool = False):
    """Load SAM3D inference pipeline from a pipeline config (e.g. pipeline.yaml).

    Mirrors the notebook Inference API:
        tag = "hf"
        config_path = f"checkpoints/{tag}/pipeline.yaml"
        inference = Inference(config_path, compile=False)

    config_path should point to a YAML that instantiates the full pipeline
    (InferencePipeline or InferencePipelinePointMap) with workspace_dir set
    to the config directory so checkpoint paths in the YAML resolve correctly.

    Returns the instantiated pipeline.
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(os.path.abspath(config_path))
    config.compile_model = compile_model
    config.rendering_engine = getattr(config, "rendering_engine", "pytorch3d")

    pipeline = instantiate(config)
    return pipeline


class SAM3DFinetuningWrapper(nn.Module):
    """
    Wraps a loaded SAM3D pipeline and adds learnable layout + shape tokens
    that are decoded by the pretrained SS decoder and pose decoder.

    Forward uses only the learnable tokens (no generator forward pass):
    tokens → SS decoder → voxel; tokens → pose decoder → R, T, S.
    All gradients flow into the tokens (and optionally into the pipeline if not frozen).
    """

    def __init__(self, pipeline, freeze_backbone: bool = True, device=None, init_data: dict | None = None, use_explicit_pose: bool = True):
        super().__init__()
        self.pipeline = pipeline
        self.device = device or next(pipeline.models["ss_generator"].parameters()).device
        self.use_explicit_pose = use_explicit_pose

        # Build learnable token dict from generator's latent_mapping (output shapes)
        ss_generator = pipeline.models["ss_generator"]
        if not hasattr(ss_generator.reverse_fn, "backbone") or not hasattr(
            ss_generator.reverse_fn.backbone, "latent_mapping"
        ):
            raise ValueError(
                "SS generator does not have backbone.latent_mapping (not an MoT model?)."
            )
        latent_mapping = ss_generator.reverse_fn.backbone.latent_mapping
        self.latent_keys = list(latent_mapping.keys())

        self.learnable_tokens = nn.ParameterDict()
        
        # Try to initialize tokens from the generator using the first frame if provided
        initial_latents = None
        if init_data is not None:
            print("[Wrapper] Initializing learnable tokens via a generator forward pass...")
            with torch.no_grad():
                img = init_data["image"].unsqueeze(0).to(self.device)
                mask = init_data["mask"].unsqueeze(0).unsqueeze(0).to(self.device)
                c2w = init_data["c2w"].unsqueeze(0).to(self.device).view(1, 16)
                fov = torch.tensor([45.0], device=self.device)
                
                # Full generator pass
                # Reconstruct RGBA numpy image for pipeline preprocessing
                image_tensor = init_data["image"].permute(1, 2, 0).cpu().numpy()
                mask_tensor = init_data["mask"].cpu().numpy()
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor[..., None]
                
                image_rgba = np.concatenate([image_tensor, mask_tensor], axis=-1)
                image_rgba = (image_rgba * 255).astype(np.uint8)
                
                # Compute pointmap if pipeline supports it
                if hasattr(self.pipeline, "compute_pointmap"):
                     pointmap_dict = self.pipeline.compute_pointmap(image_rgba)
                     pointmap = pointmap_dict["pointmap"]
                else:
                     pointmap = None
                
                # Use pipeline's preprocessing to get correct input dict structure
                ss_input_dict = self.pipeline.preprocess_image(
                    image_rgba,
                    self.pipeline.ss_preprocessor,
                    pointmap=pointmap
                )
                
                # Use default condition input mapping if not specified
                input_mapping = getattr(self.pipeline, "ss_condition_input_mapping", ["image"])
                
                condition_args, condition_kwargs = self.pipeline.get_condition_input(
                    self.pipeline.condition_embedders["ss_condition_embedder"],
                    ss_input_dict,
                    input_mapping,
                )
                
                ss_generator = self.pipeline.models["ss_generator"]
                bs = img.shape[0]
                
                # Determine latent shape
                if self.pipeline.is_mm_dit():
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                else:
                    latent_shape_dict = (bs,) + (4096, 8)

                # Sample latents
                prev_steps = ss_generator.inference_steps
                ss_generator.inference_steps = 10
                
                # We assume ss_generator handles noise creation internally given the shape
                latents = ss_generator(
                    latent_shape_dict,
                    self.device,
                    *condition_args,
                    **condition_kwargs,
                )
                
                ss_generator.inference_steps = prev_steps
                
                if self.pipeline.is_mm_dit():
                    initial_latents = latents
                else:
                     # If not mm_dit, latents is the tensor but we likely need dict for latent_keys loop
                     # Assuming mm_dit is used given the context
                     pass

        for name in self.latent_keys:
            if initial_latents is not None and name in initial_latents:
                init_val = initial_latents[name].clone()
            else:
                lat = latent_mapping[name]
                L = lat.pos_emb.shape[0]
                C = lat.input_layer.in_features
                init_val = torch.randn(1, L, C, device=self.device, dtype=torch.float32) * 0.02
            self.learnable_tokens[name] = nn.Parameter(init_val)

        if self.use_explicit_pose:
            # Explicit parameters for Object World Pose
            # Initialize close to Identity (R=I, T=0, S=1)
            self.explicit_rot_6d = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=self.device))
            self.explicit_trans = nn.Parameter(torch.zeros(3, device=self.device))
            self.explicit_scale_log = nn.Parameter(torch.zeros(3, device=self.device)) # exp(0)=1
            
            # We still need pose_decoder for type compatibility if user wants to unfreeze? 
            # But we won't use it in forward.
            self.pose_decoder = pipeline.pose_decoder
        else:
            self.pose_decoder = pipeline.pose_decoder

        self.ss_decoder = pipeline.models["ss_decoder"]

        if freeze_backbone:
            for m in [pipeline.models["ss_generator"], pipeline.models["ss_decoder"]]:
                for p in m.parameters():
                    p.requires_grad = False
            if pipeline.condition_embedders.get("ss_condition_embedder") is not None:
                for p in pipeline.condition_embedders["ss_condition_embedder"].parameters():
                    p.requires_grad = False
            # If using explicit pose, freeze pose_decoder too (it's unused anyway)
            if self.use_explicit_pose:
                if isinstance(self.pose_decoder, nn.Module):
                    for p in self.pose_decoder.parameters():
                        p.requires_grad = False

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            image: (B, 3, H, W) unused when using decoder-only path; kept for API compatibility.
            mask:  (B, 1, H, W) unused.
        Returns:
            voxel_logits: (B, 1, 64, 64, 64)
            R: (B, 3, 3), T: (B, 3), S: (B, 3)
        """
        B = image.shape[0] if image is not None else 1
        device = next(self.parameters()).device

        # Build output dict from learnable tokens (expand to batch)
        output_dict = {}
        for name in self.latent_keys:
            output_dict[name] = self.learnable_tokens[name].expand(B, -1, -1)

        # Shape → voxel grid via pretrained SS decoder
        shape_latent = output_dict["shape"]
        # Decoder expects (B, 8, 16, 16, 16)
        ss_input = (
            shape_latent.permute(0, 2, 1)
            .contiguous()
            .view(shape_latent.shape[0], 8, 16, 16, 16)
        )
        voxel_logits = self.ss_decoder(ss_input)  # (B, 1, 64, 64, 64)

        # Layout → R, T, S
        if self.use_explicit_pose:
            # Use explicit World Pose parameters (broadcast to batch)
            R = rotation_6d_to_matrix(self.explicit_rot_6d).unsqueeze(0).expand(B, -1, -1)
            T = self.explicit_trans.unsqueeze(0).expand(B, -1)
            S = torch.exp(self.explicit_scale_log).unsqueeze(0).expand(B, -1)
        else:
            # Use pose decoder
            pose_dict = self.pose_decoder(output_dict)
            from pytorch3d.transforms import quaternion_to_matrix

            quat = pose_dict["rotation"]  # (B, 4) or (4,)
            trans = pose_dict["translation"]  # (B, 3) or (3,)
            scale = pose_dict["scale"]  # (B, 3) or (1, 3)
            if quat.dim() == 1:
                quat = quat.unsqueeze(0)
            if trans.dim() == 1:
                trans = trans.unsqueeze(0)
            if scale.dim() == 1:
                scale = scale.unsqueeze(0)
            R = quaternion_to_matrix(quat)  # (B, 3, 3)
            T = trans
            S = scale.expand(R.shape[0], 3)

        return voxel_logits, R, T, S


# ============================================================================
#  Model Components (from-scratch geometry model)
# ============================================================================

class FrozenDINOv2(nn.Module):
    """DINOv2 ViT-B/14 image encoder, frozen for feature extraction."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        input_size: int = 224,
    ):
        super().__init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", model_name, verbose=False
            )
        self.embed_dim: int = self.backbone.embed_dim  # 768 for ViT-B/14
        self.input_size = input_size

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Freeze everything
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # always eval

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB in [0, 1].
        Returns:
            (B, 1 + num_patches, 768) CLS + patch tokens.
        """
        x = F.interpolate(
            x, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )
        x = (x - self.mean) / self.std
        out = self.backbone.forward_features(x)
        tokens = torch.cat(
            [out["x_norm_clstoken"].unsqueeze(1), out["x_norm_patchtokens"]], dim=1
        )
        return tokens


class CrossAttentionBlock(nn.Module):
    """Pre-norm Transformer block: self-attn → cross-attn (to image) → FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_context: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_context)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True,
            kdim=d_context, vdim=d_context,
        )
        self.norm3 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x)
        ctx = self.norm_ctx(context)
        x = x + self.cross_attn(h, ctx, ctx, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class LayoutDecoder(nn.Module):
    """Pool layout tokens → rotation (6D→3×3), translation (3), scale (3)."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = d_model * 2
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
        )
        self.rot_head = nn.Linear(hidden, 6)
        self.trans_head = nn.Linear(hidden, 3)
        self.scale_head = nn.Linear(hidden, 3)

        # Initialise to identity-like transform
        nn.init.zeros_(self.rot_head.weight)
        self.rot_head.bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )
        nn.init.zeros_(self.trans_head.weight)
        # Place the object in front of the camera to avoid near-plane clipping
        self.trans_head.bias.data.copy_(torch.tensor([0.0, 0.0, 2.0]))
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)  # exp(0) = 1

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: (B, N, d_model)
        Returns:
            R (B, 3, 3), T (B, 3), S (B, 3)
        """
        pooled = tokens.mean(dim=1)
        h = self.pool_proj(pooled)
        R = rotation_6d_to_matrix(self.rot_head(h))
        T = self.trans_head(h)
        S = torch.exp(self.scale_head(h).clamp(-4, 4))  # positive, bounded
        return R, T, S


# ============================================================================
#  Geometry Model
# ============================================================================

class GeometryModel(nn.Module):
    """
    Full Geometry Model following the architecture diagram.

    Learnable parameters optimised during training:
        * ``layout_tokens``  – shared across all views, processed through
          Transformer blocks with cross-attention to image features, then
          decoded to per-view Rotation / Translation / Scale.
        * ``shape_logits``   – a shared 3D voxel grid (D³) of occupancy
          logits, directly optimised (the "shape token" in its simplest
          form).  Sigmoid is applied at render time.

    To upgrade to the full MoT + Shape Decoder architecture, replace
    ``shape_logits`` with a ``shape_tokens`` parameter, add MoT blocks
    that process both token sets jointly, and add a 3D-CNN shape decoder
    (e.g. ``SparseStructureDecoder``).
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_blocks: int = 4,
        n_layout_tokens: int = 8,
        voxel_resolution: int = 64,
        init_voxel_radius: float = 0.25,
    ):
        super().__init__()

        # ---------- Image encoder (frozen) ----------
        self.image_encoder = FrozenDINOv2()
        d_context = self.image_encoder.embed_dim  # 768

        # ---------- Layout branch ----------
        self.layout_tokens = nn.Parameter(
            torch.randn(1, n_layout_tokens, d_model) * 0.02
        )
        self.layout_blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, n_heads, d_context) for _ in range(n_blocks)]
        )
        self.layout_decoder = LayoutDecoder(d_model)

        # ---------- Shape branch (direct voxel logits) ----------
        self.voxel_resolution = voxel_resolution
        self.shape_logits = nn.Parameter(
            self._init_sphere(voxel_resolution, init_voxel_radius)
        )

    # ---- initialisation helpers ----

    @staticmethod
    def _init_sphere(res: int, radius: float) -> torch.Tensor:
        """Create a soft sphere as initial voxel logits."""
        g = torch.linspace(-0.5, 0.5, res)
        gz, gy, gx = torch.meshgrid(g, g, g, indexing="ij")
        dist = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        logits = -(dist - radius) * 10.0  # positive inside
        return logits.unsqueeze(0).unsqueeze(0)  # (1, 1, D, D, D)

    # ---- forward ----

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            image: (B, 3, H, W) RGB in [0, 1].
            mask:  (B, 1, H, W) object mask (optional, for masked encoding).
        Returns:
            voxel_logits: (B, 1, D, D, D) raw occupancy logits.
            R: (B, 3, 3), T: (B, 3), S: (B, 3) per-view layout.
        """
        B = image.shape[0]

        # Optionally mask the image (zero background)
        if mask is not None:
            encoder_input = image * mask + 0.5 * (1 - mask)  # gray bg
        else:
            encoder_input = image

        # Image features (frozen, detached)
        with torch.no_grad():
            image_features = self.image_encoder(encoder_input)
        image_features = image_features.detach()

        # Layout tokens → transformer → decoder
        layout_h = self.layout_tokens.expand(B, -1, -1)
        for block in self.layout_blocks:
            layout_h = block(layout_h, image_features)
        R, T, S = self.layout_decoder(layout_h)

        # Shape: shared voxel logits (view-independent)
        voxel_logits = self.shape_logits.expand(B, -1, -1, -1, -1)

        return voxel_logits, R, T, S


# ============================================================================
#  Differentiable Voxel Rendering
# ============================================================================

def render_voxel_to_mask(
    voxel_logits: torch.Tensor,
    obj_to_cam: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    n_depth: int = 64,
    near: float = 0.1,
    far: float = 5.0,
) -> torch.Tensor:
    """Differentiably render a voxel grid as a 2D silhouette mask.

    For each pixel in the target camera we cast a ray, sample the
    occupancy volume at ``n_depth`` points along the ray, and take the
    **max** occupancy as the pixel value (conservative silhouette).

    Differentiable w.r.t. both ``voxel_logits`` (through ``grid_sample``)
    and ``obj_to_cam`` (through the coordinate transform of sample points).

    Args:
        voxel_logits: (B, 1, D, D, D) raw occupancy logits.
        obj_to_cam:   (B, 4, 4) object-space → target-camera (OpenCV).
        K:            (3, 3) camera intrinsics.
        H, W:         target image size.
        n_depth:      depth samples per ray.
        near, far:    depth range.

    Returns:
        (B, 1, H, W) soft silhouette in [0, 1].
    """
    B = voxel_logits.shape[0]
    device = voxel_logits.device
    dtype = voxel_logits.dtype

    occ = torch.sigmoid(voxel_logits)  # (B, 1, D, D, D)

    # Inverse transform: camera → object space
    cam_to_obj = torch.inverse(obj_to_cam)  # (B, 4, 4)

    # Pixel grid (OpenCV: z-forward, y-down, x-right)
    u_coords = torch.arange(W, device=device, dtype=dtype) + 0.5
    v_coords = torch.arange(H, device=device, dtype=dtype) + 0.5
    vv, uu = torch.meshgrid(v_coords, u_coords, indexing="ij")

    # Un-project to unit-depth rays in camera space
    x_ray = (uu - K[0, 2].to(device)) / K[0, 0].to(device)
    y_ray = (vv - K[1, 2].to(device)) / K[1, 1].to(device)

    # Depth samples
    depths = torch.linspace(near, far, n_depth, device=device, dtype=dtype)

    # 3D sample points in camera space  (n_depth, H, W, 3)
    pts_x = x_ray[None] * depths[:, None, None]
    pts_y = y_ray[None] * depths[:, None, None]
    pts_z = depths[:, None, None].expand(-1, H, W)
    pts_cam = torch.stack([pts_x, pts_y, pts_z], dim=-1)

    # Flatten → homogeneous  (N, 4)  where N = n_depth * H * W
    N = n_depth * H * W
    pts_flat = pts_cam.reshape(N, 3)
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    pts_homo = torch.cat([pts_flat, ones], dim=-1)  # (N, 4)

    # Expand for batch and transform to object space
    pts_homo = pts_homo.unsqueeze(0).expand(B, -1, -1)  # (B, N, 4)
    # (B, N, 4) @ (B, 4, 4)^T → (B, N, 4)
    pts_obj = torch.bmm(pts_homo, cam_to_obj.transpose(1, 2))[:, :, :3]

    # Normalise to [-1, 1] for F.grid_sample (voxel spans [-0.5, 0.5])
    pts_norm = pts_obj * 2.0

    # grid_sample expects (B, D_out, H_out, W_out, 3)
    # with grid[..., 0]=x → dim-4 (W), grid[..., 1]=y → dim-3 (H), grid[..., 2]=z → dim-2 (D)
    grid = pts_norm.reshape(B, n_depth, H, W, 3)

    sampled = F.grid_sample(
        occ, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )  # (B, 1, n_depth, H, W)

    # Volumetric rendering (alpha compositing) for smoother gradients
    alpha = sampled  # (B, 1, n_depth, H, W)
    
    # Calculate transmittance
    transmittance = torch.cumprod(1.0 - alpha + 1e-6, dim=2)
    # Shift transmittance to start at 1.0 (light hasn't been absorbed yet)
    ones = torch.ones_like(transmittance[:, :, :1, :, :])
    transmittance = torch.cat([ones, transmittance[:, :, :-1, :, :]], dim=2)
    
    # Rendered mask is sum of absorbed light (alpha * transmittance)
    rendered = (transmittance * alpha).sum(dim=2)  # (B, 1, H, W)
    rendered = rendered.clamp(0.0, 1.0)
    
    return rendered


# ============================================================================
#  Training
# ============================================================================

def compute_step_losses(
    model: GeometryModel,
    data_i: dict,
    data_j: dict,
    K: torch.Tensor,
    render_H: int,
    render_W: int,
    n_depth: int,
    near: float,
    far: float,
    device: torch.device,
    is_view_dependent: bool = True,
):
    """Compute cross-view and self-consistency losses for one (i, j) pair.

    Returns a dict with individual loss terms.
    """
    img_i = data_i["image"].unsqueeze(0).to(device)
    mask_i = data_i["mask"].unsqueeze(0).unsqueeze(0).to(device)
    c2w_i = data_i["c2w"].unsqueeze(0).to(device)

    mask_j = data_j["mask"].unsqueeze(0).unsqueeze(0).to(device)
    c2w_j = data_j["c2w"].unsqueeze(0).to(device)

    # Forward on view_i
    voxel_logits, R_i, T_i, S_i = model(img_i, mask_i)

    # Layout matrix: object pose
    layout_mat = build_layout_matrix(R_i, T_i, S_i)

    # ---- Cross-view loss: project to view_j ----
    w2c_j = torch.inverse(c2w_j)
    if is_view_dependent:
        # layout_mat is relative pose to view i (object → camera_i)
        obj_to_cam_j = w2c_j @ c2w_i @ layout_mat
    else:
        # layout_mat is world pose (object → world)
        obj_to_cam_j = w2c_j @ layout_mat

    rendered_j = render_voxel_to_mask(
        voxel_logits, obj_to_cam_j, K, render_H, render_W, n_depth, near, far
    )
    # Resize GT mask to render resolution
    if render_H != mask_j.shape[2] or render_W != mask_j.shape[3]:
        mask_j_rs = F.interpolate(
            mask_j, size=(render_H, render_W), mode="nearest"
        )
    else:
        mask_j_rs = mask_j

    iou_cross = soft_iou(rendered_j, mask_j_rs)
    bce_cross = F.binary_cross_entropy(
        rendered_j.clamp(1e-6, 1 - 1e-6), mask_j_rs, reduction="mean"
    )

    # ---- Self-consistency loss: project back to view_i ----
    if is_view_dependent:
        obj_to_cam_i = layout_mat  # identity relative transform
    else:
        w2c_i = torch.inverse(c2w_i)
        obj_to_cam_i = w2c_i @ layout_mat

    rendered_i = render_voxel_to_mask(
        voxel_logits, obj_to_cam_i, K, render_H, render_W, n_depth, near, far
    )
    if render_H != mask_i.shape[2] or render_W != mask_i.shape[3]:
        mask_i_rs = F.interpolate(
            mask_i, size=(render_H, render_W), mode="nearest"
        )
    else:
        mask_i_rs = mask_i

    iou_self = soft_iou(rendered_i, mask_i_rs)
    bce_self = F.binary_cross_entropy(
        rendered_i.clamp(1e-6, 1 - 1e-6), mask_i_rs, reduction="mean"
    )

    return {
        "iou_cross": iou_cross,
        "bce_cross": bce_cross,
        "iou_self": iou_self,
        "bce_self": bce_self,
        "rendered_j": rendered_j.detach(),
        "rendered_i": rendered_i.detach(),
    }


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Dataset ----
    print(f"Loading dataset from {args.blender_dir} …")
    dataset = BlenderMaskDataset(
        args.blender_dir, args.mask_pt, image_size=args.image_size
    )
    K = dataset.get_intrinsics(args.render_size, args.render_size).to(device)
    N_views = len(dataset)
    assert N_views >= 2, "Need at least 2 views for cross-view training"

    # ---- Model (from checkpoint or from scratch) ----
    use_sam3d_checkpoint = getattr(args, "config_path", None) not in (None, "")
    is_view_dependent = not use_sam3d_checkpoint

    if use_sam3d_checkpoint:
        print(f"Loading SAM3D pipeline from {args.config_path} (finetuning) …")
        pipeline = load_sam3d_pipeline(args.config_path, compile_model=False)
        pipeline.models.to(device)
        if getattr(pipeline, "condition_embedders", None):
            for emb in pipeline.condition_embedders.values():
                if emb is not None:
                    emb.to(device)
        model = SAM3DFinetuningWrapper(
            pipeline,
            freeze_backbone=args.freeze_backbone,
            device=device,
            init_data=dataset[0],
            use_explicit_pose=True, # Always use explicit pose for consistency optimization
        )
        model.to(device)
        
        # Collect params
        token_params = list(model.learnable_tokens.parameters())
        if model.use_explicit_pose:
             token_params.extend([
                 model.explicit_rot_6d,
                 model.explicit_trans,
                 model.explicit_scale_log
             ])
        token_set = {p for p in token_params}
        weight_params = [p for p in model.parameters() if p.requires_grad and p not in token_set]
    else:
        print("Building from-scratch GeometryModel …")
        model = GeometryModel(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            n_layout_tokens=args.n_layout_tokens,
            voxel_resolution=args.voxel_res,
            init_voxel_radius=args.init_radius,
        ).to(device)
        if hasattr(model, "shape_logits"):
            token_ids = {id(model.layout_tokens), id(model.shape_logits)}
            token_params = [p for p in model.parameters() if id(p) in token_ids]
            weight_params = [
                p for p in model.parameters()
                if p.requires_grad and id(p) not in token_ids
            ]
        elif hasattr(model, "learnable_tokens"):
             # For finetuning wrapper
            token_params = list(model.learnable_tokens.parameters())
            # Include explicit pose params in token_params (high LR)
            if model.use_explicit_pose:
                 token_params.extend([
                     model.explicit_rot_6d,
                     model.explicit_trans,
                     model.explicit_scale_log
                 ])
            token_set = set(token_params)
            weight_params = [p for p in model.parameters() if p.requires_grad and p not in token_set]


    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total_p:,}  trainable: {train_p:,}")
    print(f"Training mode: {'View-Dependent Pose (Scratch)' if is_view_dependent else 'World Pose (Shared Tokens)'}")

    # ---- Optimiser (separate LR for tokens vs. model weights) ----
    opt_groups = [{"params": token_params, "lr": args.lr_tokens}]
    if weight_params:
        opt_groups.append({"params": weight_params, "lr": args.lr_model})
    optimizer = torch.optim.AdamW(opt_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=args.lr_tokens * 0.01
    )

    # ---- Training loop ----
    print(f"\nStarting training for {args.num_steps} steps …\n")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for step in range(1, args.num_steps + 1):
        model.train()

        # Pick two random views
        i, j = random.sample(range(N_views), 2)
        data_i = dataset[i]
        data_j = dataset[j]

        losses = compute_step_losses(
            model, data_i, data_j, K,
            render_H=args.render_size, render_W=args.render_size,
            n_depth=args.n_depth, near=args.near, far=args.far,
            device=device,
            is_view_dependent=is_view_dependent,
        )

        # Weighted total loss
        loss = (
            (1 - losses["iou_cross"]) * args.w_iou
            + losses["bce_cross"] * args.w_bce
            + (1 - losses["iou_self"]) * args.w_iou * args.w_self
            + losses["bce_self"] * args.w_bce * args.w_self
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ---- Logging ----
        if step % args.log_every == 0 or step == 1:
            occ_str = ""
            with torch.no_grad():
                if hasattr(model, "shape_logits"):
                    occ_frac = (
                        (torch.sigmoid(model.shape_logits) > 0.5).float().mean().item()
                    )
                    occ_str = f"  occ={occ_frac:.4f}"
            print(
                f"[{step:>6d}/{args.num_steps}]  "
                f"loss={loss.item():.4f}  "
                f"IoU_cross={losses['iou_cross'].item():.4f}  "
                f"IoU_self={losses['iou_self'].item():.4f}"
                f"{occ_str}  "
                f"views={data_i['name']}→{data_j['name']}"
            )

        # ---- Checkpoints ----
        if step % args.save_every == 0 or step == args.num_steps:
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }
            if hasattr(model, "shape_logits"):
                ckpt["shape_logits"] = model.shape_logits.data.cpu()
                ckpt["layout_tokens"] = model.layout_tokens.data.cpu()
            elif hasattr(model, "learnable_tokens"):
                ckpt["learnable_tokens"] = {k: v.data.cpu() for k, v in model.learnable_tokens.items()}
                if model.use_explicit_pose:
                    ckpt["explicit_pose"] = {
                        "rot": model.explicit_rot_6d.data.cpu(),
                        "trans": model.explicit_trans.data.cpu(),
                        "scale": model.explicit_scale_log.data.cpu()
                    }
            path = os.path.join(args.output_dir, f"checkpoint_{step:06d}.pt")
            torch.save(ckpt, path)
            print(f"  → saved {path}")

            # Save voxel coordinates for quick inspection
            with torch.no_grad():
                if hasattr(model, "shape_logits"):
                    occ = torch.sigmoid(model.shape_logits.detach()) > 0.5
                else:
                    voxel_logits, _, _, _ = model(None, None)
                    occ = torch.sigmoid(voxel_logits.detach()) > 0.5
                if occ is not None:
                    coords = torch.nonzero(occ.squeeze(), as_tuple=False).cpu()
                    torch.save(
                        {"coords": coords, "resolution": occ.shape[-1]},
                        os.path.join(args.output_dir, f"voxel_{step:06d}.pt"),
                    )

        # ---- Optional: save visualisation ----
        if args.vis_every > 0 and (step % args.vis_every == 0 or step == 1):
            _save_vis(
                losses["rendered_j"],
                data_j["mask"],
                losses["rendered_i"],
                data_i["mask"],
                step,
                args.output_dir,
                data_i["name"],
                data_j["name"],
                args.render_size,
            )

    print("\n✓ Training complete.")


def _save_vis(
    rendered_j, gt_mask_j, rendered_i, gt_mask_i,
    step, output_dir, name_i, name_j, render_size,
):
    """Save a simple 2×2 grid: rendered-i | GT-i | rendered-j | GT-j."""
    try:
        rj = (rendered_j[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gj = gt_mask_j.cpu().numpy()
        if gj.ndim > 2:
            gj = gj.squeeze()
        gj_rs = np.array(
            Image.fromarray((gj * 255).astype(np.uint8)).resize(
                (render_size, render_size), Image.NEAREST
            )
        )
        ri = (rendered_i[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gi = gt_mask_i.cpu().numpy()
        if gi.ndim > 2:
            gi = gi.squeeze()
        gi_rs = np.array(
            Image.fromarray((gi * 255).astype(np.uint8)).resize(
                (render_size, render_size), Image.NEAREST
            )
        )
        top = np.concatenate([ri, gi_rs], axis=1)
        bot = np.concatenate([rj, gj_rs], axis=1)
        grid = np.concatenate([top, bot], axis=0)
        vis_path = os.path.join(output_dir, f"vis_{step:06d}.png")
        Image.fromarray(grid).save(vis_path)
    except Exception as e:
        print(f"  (vis save failed: {e})")


# ============================================================================
#  CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Multi-view 3D shape optimisation (SAM3D consistency)"
    )

    # Data
    p.add_argument("--blender_dir", type=str, required=True,
                    help="Blender dataset directory")
    p.add_argument("--mask_pt", type=str, required=True,
                    help=".pt mask dict {name: np.ndarray(bool)}")
    p.add_argument("--image_size", type=int, default=256,
                    help="Input image resolution")

    # SAM3D checkpoint (finetuning)
    p.add_argument("--config_path", type=str, default=None,
                    help="Path to pipeline config (e.g. checkpoints/hf/pipeline.yaml). If set, load SAM3D and finetune with learnable tokens.")
    p.add_argument("--freeze_backbone", action="store_true", default=True,
                    help="Freeze SS generator/decoder/condition embedder when finetuning (only train tokens).")
    p.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone",
                    help="Finetune full pipeline (tokens + backbone) with small LR.")

    # Model (used only when config_path is not set)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_blocks", type=int, default=4)
    p.add_argument("--n_layout_tokens", type=int, default=8)
    p.add_argument("--voxel_res", type=int, default=64,
                    help="Voxel grid resolution (D³)")
    p.add_argument("--init_radius", type=float, default=0.25,
                    help="Initial sphere radius for shape logits")

    # Rendering
    p.add_argument("--render_size", type=int, default=128,
                    help="Resolution for differentiable rendering")
    p.add_argument("--n_depth", type=int, default=64,
                    help="Depth samples per ray")
    p.add_argument("--near", type=float, default=0.1)
    p.add_argument("--far", type=float, default=5.0)

    # Training
    p.add_argument("--num_steps", type=int, default=5000)
    p.add_argument("--lr_tokens", type=float, default=1e-3,
                    help="LR for layout tokens & shape logits")
    p.add_argument("--lr_model", type=float, default=1e-4,
                    help="LR for transformer / decoder weights")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--w_iou", type=float, default=1.0, help="IoU loss weight")
    p.add_argument("--w_bce", type=float, default=0.5, help="BCE loss weight")
    p.add_argument("--w_self", type=float, default=0.5,
                    help="Self-consistency loss multiplier")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/sam3d_consistent")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--vis_every", type=int, default=200,
                    help="Save mask visualisations (0 = off)")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
