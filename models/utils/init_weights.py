import torch


def initialize_weights_and_bias(m):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.0)


def init_weights(m):
    if isinstance(m, torch.nn.ParameterList):
        for param in m:
            torch.nn.init.xavier_uniform_(param.data)

    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        initialize_weights_and_bias(m)
