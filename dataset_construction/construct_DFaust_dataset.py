import os
import numpy as np
import torch
import json
from pathlib import Path

from utils import (
    get_device,
    load_eigenvectors,
    normalize_vertices,
    SMPLModelCache,
    find_dataset_path,
    process_poses,
    prepare_smplx_batch,
    project_to_eigenbasis,
    save_dataset_split
)

# Device
comp_device = get_device()
print(f"Using device: {comp_device}")

# Load Graph Laplacian eigenvectors
nb_freq = 4096  # Number of basis functions to use
evecs = load_eigenvectors(nb_freq)

# Construct only DFaust dataset
dataset_exp = "only_DFaust"

datasets = [
    "DFaust"
]

# SMPL model configuration
smpl_model_dir = 'data/SMPL/smplx'
num_betas = 16  # number of body parameters

# Initialize SMPL model cache
smpl_cache = SMPLModelCache(smpl_model_dir, num_betas)

# Create dataset output directory
datasets_path = Path("data/SMPL/Datasets")
datasets_path.mkdir(parents=True, exist_ok=True)

# Base path for AMASS data
base_path = Path("data/SMPL/AMASS")

# Count identities (folders that directly contain .npz files)
identities = set()
total_frames = 0
npz_file_count = 0

# Storage for train and vald data
train_data = []  # List of (infos, coeffs) tuples
vald_data = []   # List of (infos, coeffs, vertices) tuples
identity_data = {}  # Dictionary to store frames by identity

# Browse each dataset
for dataset_name in datasets:
    try:
        dataset_path = find_dataset_path(base_path, dataset_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(dataset_path):
        # Check if this directory contains .npz files
        has_npz = any(file.endswith('.npz') and file != 'shape.npz' for file in files)
        
        if has_npz:
            # This folder contains .npz files, so it's an identity
            # Get just the folder name (last part of the path)
            folder_name = os.path.basename(root)
            identities.add(folder_name)
        
        # Process all npz files
        for file in files:
            if file.endswith('.npz') and file != 'shape.npz':
                npz_path = os.path.join(root, file)
                try:
                    bdata = np.load(npz_path, allow_pickle=True)
                    
                    # Check for required keys
                    if 'poses' not in bdata or 'trans' not in bdata:
                        print(f"  Warning: Missing 'poses' or 'trans' in {npz_path}")
                        continue
                    
                    # Get gender
                    subject_gender = str(bdata['gender']) if 'gender' in bdata else 'neutral'
                    # This script only instantiates male/female SMPLH models
                    if subject_gender not in ('male', 'female'):
                        subject_gender = 'male'
                    
                    # Get number of frames
                    time_length = len(bdata['trans'])
                    num_frames = time_length
                    total_frames += num_frames
                    npz_file_count += 1
                    
                    print(f"  Processing {file}: {num_frames} frames (gender: {subject_gender})")

                    # Get SMPL model for this gender
                    bm = smpl_cache.get(subject_gender)

                    # Prepare pose parameters for all frames
                    poses_np = bdata['poses']
                    root_orient, body_pose, left_hand_pose, right_hand_pose = process_poses(poses_np, comp_device)
                    
                    # Get betas (shape parameters)
                    if 'betas' in bdata:
                        betas = torch.Tensor(
                            np.repeat(bdata['betas'][:num_betas][np.newaxis], 
                                    repeats=time_length, axis=0)
                        ).to(comp_device)
                    else:
                        betas = torch.zeros((time_length, num_betas)).to(comp_device)
                    
                    trans = torch.Tensor(bdata['trans']).to(comp_device)
                    
                    # Get identity folder name for train/vald split
                    folder_name = os.path.basename(root)
                    
                    # Process frames in batches to avoid memory issues
                    batch_size = 2048
                    for batch_start in range(0, time_length, batch_size):
                        batch_end = min(batch_start + batch_size, time_length)
                        
                        # Prepare batch parameters
                        batch_root_orient = root_orient[batch_start:batch_end]
                        zero_transl = torch.zeros_like(trans[batch_start:batch_end])
                        smplx_params = prepare_smplx_batch(
                            batch_root_orient,
                            body_pose[batch_start:batch_end],
                            betas[batch_start:batch_end],
                            zero_transl,
                            left_hand_pose[batch_start:batch_end] if left_hand_pose is not None else None,
                            right_hand_pose[batch_start:batch_end] if right_hand_pose is not None else None,
                            comp_device,
                        )

                        # Forward pass through SMPL model
                        with torch.no_grad():
                            body_output = bm(**smplx_params)
                            batch_verts = body_output.vertices  # (B,V,3)
                            batch_joints = body_output.joints   # (B,J,3)

                        # Normalize: pelvis joint at origin and consistent orientation
                        batch_verts_norm = normalize_vertices(
                            batch_verts,
                            batch_joints,
                            batch_root_orient,
                        )
                        
                        # Project normalized vertices onto eigenbasis
                        batch_coeffs = project_to_eigenbasis(batch_verts_norm, evecs)
                        
                        # Store data for each frame in the batch
                        for i in range(batch_verts_norm.shape[0]):
                            frame_idx = batch_start + i
                            
                            # Create infos dictionary
                            infos = {
                                "identity": folder_name,
                                "dataset": dataset_name,
                                "file": file,
                                "frame_idx": frame_idx,
                                "gender": subject_gender
                            }
                            
                            # Get coefficients for this frame (nb_freq, 3)
                            frame_coeffs = batch_coeffs[i].cpu()
                            
                            # Store based on identity (will determine train/vald split later)
                            frame_data = {
                                "infos": infos,
                                "coeffs": frame_coeffs,
                                "vertices": batch_verts_norm[i].cpu(),  # Normalized vertices for eval set
                            }
                            
                            # Store by identity for later splitting
                            if folder_name not in identity_data:
                                identity_data[folder_name] = []
                            identity_data[folder_name].append(frame_data)
                    
                except Exception as e:
                    print(f"  Error processing {npz_path}: {e}")
                    import traceback
                    traceback.print_exc()

# Get all identities from the processed dataset
identities_list = sorted(list(identities))

# DFaust split: 8 first identities for training, 2 last for eval
train_identities = identities_list[:8]
eval_identities = identities_list[-2:]

print(f"\n{'='*60}")
print(f"Splitting data for {dataset_exp}")
print(f"{'='*60}")
print(f"Train identities: {train_identities}")
print(f"Eval identities: {eval_identities}")

# Split data into train and vald sets
nb_train = 0
nb_eval = 0

for identity, frames in identity_data.items():
    if identity in train_identities:
        train_data.extend(frames)
        nb_train += len(frames)
    elif identity in eval_identities:
        vald_data.extend(frames)
        nb_eval += len(frames)

print(f"Number of train frames: {nb_train}")
print(f"Number of eval frames: {nb_eval}")

# Create dataset output folder
dataset_output_path = datasets_path / dataset_exp
dataset_output_path.mkdir(parents=True, exist_ok=True)

# Save split info to JSON
split_info = {
    "nb_train": nb_train,
    "nb_evals": nb_eval,
    "train_identities": train_identities,
    "eval_identities": eval_identities
}

json_path = dataset_output_path / "infos.json"
with open(json_path, 'w') as f:
    json.dump(split_info, f, indent=4)

print(f"\n✓ Split info saved to {json_path}")


# Store coeffs as big numpy arrays and save in .npy format for memmap
print(f"\nConverting train and vald coeffs to numpy arrays...")
train_coeffs = np.stack([frame['coeffs'].numpy() for frame in train_data], axis=0)
vald_coeffs = np.stack([frame['coeffs'].numpy() for frame in vald_data], axis=0)

# Store infos and vertices separately
train_infos = [frame['infos'] for frame in train_data]
vald_infos = [frame['infos'] for frame in vald_data]
vald_vertices = np.stack([frame['vertices'].numpy() for frame in vald_data], axis=0)

# Save dataset using utility function
save_dataset_split(
    dataset_output_path,
    train_coeffs,
    vald_coeffs,
    train_infos,
    vald_infos,
    vald_vertices,
    split_info
)

print(f"\n{'='*60}")
print(f"Dataset {dataset_exp} Complete!")
print(f"{'='*60}")
print(f"Output directory: {dataset_output_path}")
print(f"Files created:")
print(f"  - infos.json")
print(f"  - train_coeffs.npy (coeffs: {train_coeffs.shape})")
print(f"  - vald_coeffs.npy (coeffs: {vald_coeffs.shape})")
print(f"  - vald_vertices.npy (vertices: {vald_vertices.shape})")
print(f"  - train_infos.json")
print(f"  - vald_infos.json")
