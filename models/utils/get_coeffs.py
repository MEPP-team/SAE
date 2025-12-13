import torch
import time
from train import get_dataloader


def get_coeffs(wanted_index, opt, dataset_type):
    dataloader = opt['dataloader_' + dataset_type]

    coeffs = torch.zeros((1, opt['nb_freq'], 3)).to(opt["device"])

    if dataset_type == 'vald':
        vertices = torch.zeros((1, opt['nb_vertices'], 3)).to(opt["device"])
    else:
        vertices = None

    if dataset_type == 'vald':
        _, coeffs_dataloader, vertices_dataloader = dataloader.dataset.__getitem__(wanted_index)

        coeffs[0, ...] = coeffs_dataloader[:opt['nb_freq'], ...]
        vertices[0, ...] = vertices_dataloader

    elif dataset_type == 'train':
        _, coeffs_dataloader = dataloader.dataset.__getitem__(wanted_index)

        coeffs[0, ...] = coeffs_dataloader[:opt['nb_freq'], ...]
    else:
        print('Dataset type not recognized.')
        exit()

    coeffs = coeffs.to(opt['device']).float()

    return coeffs, vertices
