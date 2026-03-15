# scripts/utils_smplx.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import smplx


def to_tensor(x, device: str, dtype=torch.float32):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    x = np.asarray(x)
    return torch.tensor(x, device=device, dtype=dtype)


def ensure_batch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    return x


def load_smplx_model(
    model_root: str | Path,
    gender: str = "neutral",
    device: str = "cuda",
    num_betas: int = 10,
    use_pca: bool = False,
    flat_hand_mean: bool = True,
    kid_template_path: Optional[str | Path] = None,
):
    create_kwargs = dict(
        model_path=str(model_root),
        model_type="smplx",
        gender=gender,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        num_betas=num_betas,
        ext="npz",
    )

    # kid template가 있으면 age='kid'로 생성
    if kid_template_path is not None:
        create_kwargs["age"] = "kid"
        create_kwargs["kid_template_path"] = str(kid_template_path)

    model = smplx.create(**create_kwargs)
    return model.to(device)


def build_canonical_output(
    model,
    params: Dict[str, Any],
    device: str = "cuda",
):
    """
    shape는 유지하고, pose 관련 값은 모두 0으로 둔 canonical body 생성
    """
    if "betas" not in params:
        raise KeyError("params must contain 'betas'")

    betas = ensure_batch(params["betas"])
    betas = to_tensor(betas, device)

    batch_size = betas.shape[0]

    def zeros(shape):
        return torch.zeros(shape, dtype=torch.float32, device=device)

    # expression 차원은 모델에 따라 다를 수 있어서 안전하게 처리
    num_expression_coeffs = getattr(model, "num_expression_coeffs", 10)
    num_body_joints = getattr(model, "NUM_BODY_JOINTS", 21)
    num_hand_joints = getattr(model, "NUM_HAND_JOINTS", 15)

    output = model(
        betas=betas,
        transl=zeros((batch_size, 3)),
        global_orient=zeros((batch_size, 3)),
        body_pose=zeros((batch_size, num_body_joints * 3)),
        left_hand_pose=zeros((batch_size, num_hand_joints * 3)),
        right_hand_pose=zeros((batch_size, num_hand_joints * 3)),
        jaw_pose=zeros((batch_size, 3)),
        leye_pose=zeros((batch_size, 3)),
        reye_pose=zeros((batch_size, 3)),
        expression=zeros((batch_size, num_expression_coeffs)),
        return_verts=True,
    )

    verts = output.vertices[0].detach().cpu().numpy()
    joints = output.joints[0].detach().cpu().numpy()
    faces = model.faces.astype(np.int32)

    return verts, joints, faces


def center_by_pelvis(
    verts: np.ndarray,
    joints: np.ndarray,
    pelvis_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    pelvis = joints[pelvis_idx].copy()
    verts = verts - pelvis[None, :]
    joints = joints - pelvis[None, :]
    return verts, joints