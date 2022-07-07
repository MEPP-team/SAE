import torch


def error_meshes_mm(a, b):
    distances = torch.sqrt(torch.sum((a - b) ** 2, dim=2))

    return distances
