import os
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm

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

# Use AMASS splits provided by the user
dataset_exp = "AMASS_splits"

amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT',
              'DanceDB', 'BMLhandball', 'Transitions_mocap', 'EKUT', 'TCD_handMocap', 'ACCAD']
}

# Flatten datasets to process (unique)
datasets = []
for v in amass_splits.values():
    for d in v:
        if d not in datasets:
            datasets.append(d)

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

# Storage for train/vald data
train_data = []
vald_data = []
total_frames = 0
npz_file_count = 0

# Preinit each body model
bm_male = smpl_cache.get('male')
bm_female = smpl_cache.get('female')

# Skin per batch
batch_size = 4096
all_smplx_params_male = None
all_smplx_params_female = None
all_smplx_params_neutral = None
all_infos_male = None
all_infos_female = None
all_infos_neutral = None

all_params_and_infos = {}

def process_smplx_batch(gender):
    global all_smplx_params_male, all_smplx_params_female, all_smplx_params_neutral
    global all_infos_male, all_infos_female, all_infos_neutral
    global train_data, vald_data

    # Forward pass through SMPL model
    with torch.no_grad():
        if gender == 'male':
            bm = bm_male
            all_smplx_params = all_smplx_params_male
            all_infos = all_infos_male
        elif gender == 'female':
            bm = bm_female
            all_smplx_params = all_smplx_params_female
            all_infos = all_infos_female
        else:
            raise ValueError(
                "Neutral gender encountered but neutral SMPLH model is not initialized. "
                "Either enable bm_neutral, or map neutral to male/female upstream."
            )

        if all_smplx_params is None or all_infos is None:
            return

        # Ensure all tensors share the same batch size
        batch_n = all_smplx_params['body_pose'].shape[0]
        for key, tensor in all_smplx_params.items():
            if not torch.is_tensor(tensor):
                continue
            if tensor.shape[0] != batch_n:
                raise RuntimeError(
                    f"Batch size mismatch for '{key}': expected {batch_n}, got {tensor.shape[0]}"
                )

        # SMPLH expects hand poses; default to zeros if not provided
        device = all_smplx_params['body_pose'].device
        dtype = all_smplx_params['body_pose'].dtype
        all_smplx_params.setdefault('left_hand_pose', torch.zeros((batch_n, 45), device=device, dtype=dtype))
        all_smplx_params.setdefault('right_hand_pose', torch.zeros((batch_n, 45), device=device, dtype=dtype))

        body_output = bm(**all_smplx_params)
        batch_verts = body_output.vertices  # (B,V,3)
        batch_joints = body_output.joints   # (B,J,3)

        # Normalize vertices: pelvis at origin + canonical orientation
        batch_root_orient = all_smplx_params.get('root_orient_for_norm', None)
        if batch_root_orient is None:
            raise RuntimeError("Missing 'root_orient_for_norm' in batch params; cannot normalize orientation")
        batch_verts = normalize_vertices(batch_verts, batch_joints, batch_root_orient)
    
    # Project vertices onto eigenbasis
    batch_coeffs = project_to_eigenbasis(batch_verts, evecs)
    
    # Store data for each frame in the batch
    for i in range(batch_verts.shape[0]):
        # Create infos dictionary
        infos = {
            "identity": all_infos['identity'][i],
            "dataset": all_infos['dataset'][i],
            "file": all_infos['file'][i],
            "frame_idx": all_infos['frame_idx'][i],
            "gender": all_infos['gender'][i]
        }

        if infos['dataset'] in amass_splits.get('train', []):
            # Store frame data and route to split according to dataset membership
            frame_data = {
                "infos": infos,
                "coeffs": batch_coeffs[i].cpu(),
            }

            train_data.append(frame_data)
        elif infos['dataset'] in amass_splits.get('vald', []):
            # Store frame data and route to split according to dataset membership
            frame_data = {
                "infos": infos,
                "coeffs": batch_coeffs[i].cpu(),
                "vertices": batch_verts[i].cpu(),
            }

            vald_data.append(frame_data)
        else:
            # If dataset not listed, default to train and warn
            print(f"  Warning: dataset '{infos['dataset']}' not in amass_splits — defaulting to train")
            train_data.append(frame_data)

    # Reset batch storage
    if gender == 'male':
        all_smplx_params_male = None
        all_infos_male = None
    elif gender == 'female':
        all_smplx_params_female = None
        all_infos_female = None
    else:
        all_smplx_params_neutral = None
        all_infos_neutral = None

    # Clear GPU memory
    torch.cuda.empty_cache()


 # Browse each dataset
for dataset_name in tqdm(datasets, desc="Datasets", unit="dataset", leave=False):
    try:
        dataset_path = find_dataset_path(base_path, dataset_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

    # Walk through all subdirectories
    # Count total directories to set tqdm total
    all_dirs = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        all_dirs.append(dirpath)

    # tqdm progress bar for directories
    for root, dirs, files in tqdm(list(os.walk(dataset_path)), desc=f"{dataset_name} dirs", unit="dir", leave=False):
        # Check if this directory contains .npz files
        has_npz = any(file.endswith('.npz') and file != 'shape.npz' for file in files)

        # Process all npz files with tqdm progress bar
        for file in tqdm(files, desc=f"{dataset_name} files in {os.path.basename(root)}", unit="file", leave=False):
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
                    
                    # 1 out of 100 frames from middle 90% of the sequence
                    start_idx = int(num_frames * 0.10)
                    end_idx = int(num_frames * 0.90)
                    indices_to_process = list(range(start_idx, end_idx, 100))
                    indices_batch = indices_to_process

                    # Prepare batch parameters
                    batch_root_orient = root_orient[indices_batch]
                    zero_transl = torch.zeros_like(trans[indices_batch])
                    smplx_params = prepare_smplx_batch(
                        batch_root_orient,
                        body_pose[indices_batch],
                        betas[indices_batch],
                        zero_transl,
                        left_hand_pose[indices_batch] if left_hand_pose is not None else None,
                        right_hand_pose[indices_batch] if right_hand_pose is not None else None,
                        comp_device,
                    )
                    
                    # Create infos dictionary
                    infos = {
                        "identity": [folder_name]*len(indices_batch),
                        "dataset": [dataset_name]*len(indices_batch),
                        "file": [file]*len(indices_batch),
                        "frame_idx": indices_batch,
                        "gender": [subject_gender]*len(indices_batch)
                    }

                    if subject_gender == 'male':

                        if all_smplx_params_male is None:
                            all_smplx_params_male = smplx_params
                        else:
                            for key in smplx_params:
                                all_smplx_params_male[key] = torch.cat(
                                    (all_smplx_params_male[key], smplx_params[key]), dim=0)

                        if all_infos_male is None:
                            all_infos_male = infos
                        else:
                            for key in infos:
                                all_infos_male[key].extend(infos[key])

                        if all_smplx_params_male['body_pose'].shape[0] >= batch_size:
                            process_smplx_batch(subject_gender)

                    elif subject_gender == 'female':

                        if all_smplx_params_female is None:
                            all_smplx_params_female = smplx_params
                        else:
                            for key in smplx_params:
                                all_smplx_params_female[key] = torch.cat(
                                    (all_smplx_params_female[key], smplx_params[key]), dim=0)

                        if all_infos_female is None:
                            all_infos_female = infos
                        else:
                            for key in infos:
                                all_infos_female[key].extend(infos[key])

                        if all_smplx_params_female['body_pose'].shape[0] >= batch_size:
                            process_smplx_batch(subject_gender)
                    else:

                        if all_smplx_params_neutral is None:
                            all_smplx_params_neutral = smplx_params
                        else:
                            for key in smplx_params:
                                all_smplx_params_neutral[key] = torch.cat(
                                    (all_smplx_params_neutral[key], smplx_params[key]), dim=0)

                        if all_infos_neutral is None:
                            all_infos_neutral = infos
                        else:
                            for key in infos:
                                all_infos_neutral[key].extend(infos[key])

                        if all_smplx_params_neutral['body_pose'].shape[0] >= batch_size:
                            process_smplx_batch(subject_gender)
                    
                except Exception as e:
                    print(f"  Error processing {npz_path}: {e}")
                    import traceback
                    traceback.print_exc()

# Process any remaining frames
if all_smplx_params_male is not None:
    process_smplx_batch('male')
if all_smplx_params_female is not None:
    process_smplx_batch('female')
if all_smplx_params_neutral is not None:
    process_smplx_batch('neutral')

# Compute counts and create output folder
nb_train = len(train_data)
nb_vald = len(vald_data)

print(f"\n{'='*60}")
print(f"Splitting data for {dataset_exp}")
print(f"{'='*60}")
print(f"Train datasets: {amass_splits.get('train', [])}")
print(f"Vald datasets: {amass_splits.get('vald', [])}")
print(f"Number of train frames: {nb_train}")
print(f"Number of vald frames: {nb_vald}")

# Create dataset output folder
dataset_output_path = datasets_path / dataset_exp
dataset_output_path.mkdir(parents=True, exist_ok=True)

# Save split info to JSON
split_info = {
    "nb_train": nb_train,
    "nb_evals": nb_vald,
    "train_datasets": amass_splits.get('train', []),
    "vald_datasets": amass_splits.get('vald', [])
}

json_path = dataset_output_path / "infos.json"
with open(json_path, 'w') as f:
    json.dump(split_info, f, indent=4)

print(f"\nSplit info saved to {json_path}")

# Store coeffs as numpy arrays and save in .npy format for memmap
print(f"\nConverting train and vald coeffs to numpy arrays...")

def stack_or_empty(list_of_frames, name):
    if len(list_of_frames) > 0:
        return np.stack([frame['coeffs'].numpy() for frame in list_of_frames], axis=0)
    else:
        return np.zeros((0, nb_freq, 3), dtype=np.float32)

def stack_vertices_or_empty(list_of_frames):
    if len(list_of_frames) > 0:
        return np.stack([frame['vertices'].numpy() for frame in list_of_frames], axis=0)
    else:
        return np.zeros((0, evecs.shape[0], 3), dtype=np.float32)

train_coeffs = stack_or_empty(train_data, 'train')
vald_coeffs = stack_or_empty(vald_data, 'vald')

# Store infos and vertices separately
train_infos = [frame['infos'] for frame in train_data]
vald_infos = [frame['infos'] for frame in vald_data]
vald_vertices = stack_vertices_or_empty(vald_data)

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
