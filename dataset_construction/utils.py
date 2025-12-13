"""
Common utilities for dataset construction scripts.
"""

import os
import json
import numpy as np
import torch
import smplx
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Device setup
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device():
    return comp_device


def load_eigenvectors(nb_freq: int, base_path: str = "data/SMPL") -> torch.Tensor:
    npy_path = Path(base_path) / "evecs_GL_6890.npy"
    txt_path = Path(base_path) / "evecs_GL_6890.txt"
    
    if not npy_path.exists():
        print(f"Loading eigenvectors from {txt_path}...")
        evecs = np.loadtxt(txt_path)
        print(f"Saving eigenvectors to {npy_path}...")
        np.save(npy_path, evecs)
    else:
        print(f"Loading eigenvectors from {npy_path}...")
        evecs = np.load(npy_path)
    
    # Load as torch tensor and select first nb_freq frequencies
    evecs = torch.from_numpy(np.load(npy_path)).float().to(comp_device)[:, :nb_freq]
    print(f"Eigenvectors shape: {evecs.shape}")
    
    return evecs


def _aa_to_rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / angle
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    ca = torch.cos(angle[:, 0])
    sa = torch.sin(angle[:, 0])
    one_m_ca = 1.0 - ca

    r00 = ca + x * x * one_m_ca
    r01 = x * y * one_m_ca - z * sa
    r02 = x * z * one_m_ca + y * sa

    r10 = y * x * one_m_ca + z * sa
    r11 = ca + y * y * one_m_ca
    r12 = y * z * one_m_ca - x * sa

    r20 = z * x * one_m_ca - y * sa
    r21 = z * y * one_m_ca + x * sa
    r22 = ca + z * z * one_m_ca

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def normalize_vertices(
    vertices: torch.Tensor,
    joints: torch.Tensor,
    root_orient_axis_angle: torch.Tensor,
) -> torch.Tensor:
    # Center on pelvis
    pelvis = joints[:, 0:1, :]  # (B, 1, 3)
    v = vertices - pelvis
    
    # Canonicalize global facing: apply inverse of root orientation
    R = _aa_to_rotmat(root_orient_axis_angle)  # (B, 3, 3)
    R_inv = R.transpose(-1, -2)
    v = torch.einsum('bij,bvj->bvi', R_inv, v)
    
    return v


class SMPLModelCache:
    def __init__(self, smpl_model_dir: str, num_betas: int = 16):
        self.smpl_model_dir = smpl_model_dir
        self.num_betas = num_betas
        self.models = {}
    
    def get(self, gender: str) -> smplx.SMPLH:
        if gender not in self.models:
            print(f"  Loading SMPL model for gender: {gender}")
            self.models[gender] = smplx.create(
                self.smpl_model_dir,
                model_type='smplh',
                gender=gender,
                num_betas=self.num_betas,
                use_pca=False,
                batch_size=1,
                ext='pkl'
            ).to(comp_device)
        return self.models[gender]


def find_dataset_path(base_path: Path, dataset_name: str) -> Path:
    direct_path = base_path / dataset_name
    if direct_path.exists():
        return direct_path
    
    # Look for subdirectories
    for subdir in base_path.iterdir():
        potential_path = subdir / dataset_name
        if potential_path.exists():
            return potential_path
    
    raise FileNotFoundError(f"Dataset {dataset_name} not found in {base_path}")


def process_poses(poses_np: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    pose_dim = poses_np.shape[1]
    
    root_orient = torch.Tensor(poses_np[:, :3]).to(device)
    
    if pose_dim >= 156:
        # SMPLH: 3 global + 63 body + 45 lh + 45 rh = 156
        body_pose = torch.Tensor(poses_np[:, 3:66]).to(device)
        left_hand_pose = torch.Tensor(poses_np[:, 66:111]).to(device)
        right_hand_pose = torch.Tensor(poses_np[:, 111:156]).to(device)
    elif pose_dim >= 72:
        # SMPL: 3 global + 69 body = 72
        body_pose = torch.Tensor(poses_np[:, 3:66]).to(device)
        left_hand_pose = None
        right_hand_pose = None
    else:
        raise ValueError(f"Unexpected pose dimension {pose_dim}")
    
    return root_orient, body_pose, left_hand_pose, right_hand_pose


def prepare_smplx_batch(
    root_orient: torch.Tensor,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    trans: torch.Tensor,
    left_hand_pose: Optional[torch.Tensor] = None,
    right_hand_pose: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> Dict:
    if device is None:
        device = comp_device
    
    batch_n = root_orient.shape[0]
    
    smplx_params = {
        'global_orient': root_orient,
        'body_pose': body_pose,
        'betas': betas,
        'transl': trans,
        'root_orient_for_norm': root_orient.clone(),  # For post-normalization
    }
    
    # Handle hand poses
    if left_hand_pose is not None and right_hand_pose is not None:
        smplx_params['left_hand_pose'] = left_hand_pose
        smplx_params['right_hand_pose'] = right_hand_pose
    else:
        smplx_params['left_hand_pose'] = torch.zeros((batch_n, 45), device=device)
        smplx_params['right_hand_pose'] = torch.zeros((batch_n, 45), device=device)
    
    return smplx_params


def project_to_eigenbasis(
    vertices: torch.Tensor,
    evecs: torch.Tensor
) -> torch.Tensor:
    return torch.einsum('vf,bvc->bfc', evecs, vertices)


def save_dataset_split(
    output_path: Path,
    train_coeffs: np.ndarray,
    vald_coeffs: np.ndarray,
    train_infos: List[Dict],
    vald_infos: List[Dict],
    vald_vertices: Optional[np.ndarray] = None,
    split_info: Optional[Dict] = None,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save coefficient arrays
    np.save(output_path / "train_coeffs.npy", train_coeffs)
    np.save(output_path / "vald_coeffs.npy", vald_coeffs)
    
    if vald_vertices is not None:
        np.save(output_path / "vald_vertices.npy", vald_vertices)
    
    # Save info JSON files
    with open(output_path / "train_infos.json", 'w') as f:
        json.dump(train_infos, f, indent=4)
    
    with open(output_path / "vald_infos.json", 'w') as f:
        json.dump(vald_infos, f, indent=4)
    
    # Save split info if provided
    if split_info is not None:
        with open(output_path / "infos.json", 'w') as f:
            json.dump(split_info, f, indent=4)
