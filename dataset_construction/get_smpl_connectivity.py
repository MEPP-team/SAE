import os
import smplx
import torch
import numpy as np


smpl_model_dir = 'data\SMPL\smplx'

subject_gender = 'female'
num_betas = 16
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bm = smplx.create(smpl_model_dir, model_type='smplh',
                  gender=subject_gender, num_betas=num_betas,
                  batch_size=1, ext='pkl').to(comp_device)

faces = bm.faces.astype(np.int32)

print(f"Faces shape: {faces.shape}")

faces_file = 'data/SMPL/smpl_faces.txt'
np.savetxt(faces_file, faces, fmt='%d')

print(f"Saved SMPL faces to {faces_file}")
