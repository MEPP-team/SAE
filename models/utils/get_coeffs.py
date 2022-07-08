import torch
import time
from train import get_dataloader


def get_coeffs(wanted_index, opt, dataset_type):
    start = time.time()

    dataloader = get_dataloader(opt, dataset_type, 1, wanted_index=wanted_index)

    coeffs = torch.zeros((1, opt['nb_freq'], 3)).to(opt["device"])

    if dataset_type == 'test':
        vertices = torch.zeros((1, opt['nb_vertices'], 3)).to(opt["device"])
    else:
        vertices = None

    print('Looking for mesh in dataset...', end=" ")

    # dataloader is sliced, first iteration is the wanted index
    for data in dataloader:
        coeffs[0, ...] = data[1][0, :opt['nb_freq'], ...]

        if dataset_type == 'test':
            vertices[0, ...] = data[2][0, ...]

        break

    coeffs = coeffs.to(opt['device']).float()

    end = time.time()
    print(" done. Elapsed time: ", end - start, "s")

    return coeffs, vertices
