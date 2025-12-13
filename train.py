import argparse
import os
import numpy as np
import torch
import json
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from models.utils.init_weights import init_weights


def get_opt(job_id, load):
    if load:
        print('Loading options with job_id')
        with open('checkpoints/' + str(job_id) + '/infos.json', "r") as outfile:
            opt = json.load(outfile)
    else:
        print('Loading default options for new job')

        with open('default_options.json', "r") as outfile:
            opt = json.load(outfile)

        parser = argparse.ArgumentParser()

        parser.add_argument('--job_id', required=True)

        # so we can pass other default options as program argument
        for key, value in opt.items():
            parser.add_argument('--' + key, default=value, type=type(value))

        opt = vars(parser.parse_args())

        # If checkpoints folder does not exist, create it
        if not os.path.exists('checkpoints/'):
            os.mkdir('checkpoints/')

        try:
            os.mkdir('checkpoints/' + str(opt['job_id']))
        except:
            print('Folder for job ' + str(opt['job_id']) + ' already exists.')
            exit()

        with open('checkpoints/' + str(opt['job_id']) + '/infos.json', "w") as outfile:
            json.dump(opt, outfile, sort_keys=True, indent=4)

    print('Options:\n', json.dumps(opt, sort_keys=True, indent=4), end="\n\n")

    return opt


# Custom PyTorch Dataset for compressed numpy arrays
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, coeffs, infos, vertices=None):
        self.coeffs = coeffs
        self.infos = infos
        self.vertices = vertices
    def __len__(self):
        return len(self.coeffs)
    def __getitem__(self, idx):
        coeff = torch.from_numpy(self.coeffs[idx]).float()
        info = self.infos[idx]
        if self.vertices is not None:
            vert = torch.from_numpy(self.vertices[idx]).float()
            return info, coeff, vert
        else:
            return info, coeff


def get_dataloader(opt, dataset_type, batch_size, shuffle=True):
    # Load arrays with np.memmap and infos from JSON
    coeffs_path = os.path.join(opt['path'], f"{dataset_type}_coeffs.npy")
    infos_path = os.path.join(opt['path'], f"{dataset_type}_infos.json")
    coeffs = np.memmap(coeffs_path, dtype='float32', mode='r')
    # Infer shape from file size and expected dimensions
    with open(infos_path, 'r') as f:
        infos = json.load(f)

    # Get shape from infos or opt
    nb_samples = len(infos)
    nb_freq = opt['nb_freq']
    coeffs = coeffs[32:].reshape((nb_samples, nb_freq, 3))

    if dataset_type == 'vald':
        vertices_path = os.path.join(opt['path'], f"vald_vertices.npy")
        vertices = np.memmap(vertices_path, dtype='float32', mode='r')
        nb_vertices = opt.get('nb_vertices', 6890)
        vertices = vertices[32:].reshape((nb_samples, nb_vertices, 3))
    else:
        vertices = None

        # Set mean and std for standardization (computed on training set)
        opt['mean'] = torch.tensor(coeffs.mean(axis=0)).cuda().unsqueeze(0)  # [nb_freq, 3]
        opt['std'] = torch.tensor(coeffs.std(axis=0)).cuda().unsqueeze(0)  # [nb_freq, 3]

    dataset = NumpyDataset(coeffs, infos, vertices)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt["num_workers"],
        pin_memory=True,
        persistent_workers=True if opt["num_workers"] > 0 else False
    )


def load_trainer(job_id=None, profiler="simple", seed_everything_flag=True):
    if job_id is None:
        load = False
    else:
        load = True

    torch.set_float32_matmul_precision('high')  # or 'medium'
    if seed_everything_flag:
        seed_everything(42, workers=True)
    print()

    opt = get_opt(job_id, load)

    with open(opt['path'] + '/infos.json', "r") as f:
        opt_dataset = json.load(f)

    for key, value in opt_dataset.items():
        if key in opt:
            continue
        opt[key] = value

    if opt['loss_type'] == "MSE":
        opt['loss'] = torch.nn.MSELoss()
    else:
        print('Loss type not implemented.')
        exit()

    opt['nb_vertices'] = 6890

    # get path from which the script in launched
    path_repo = os.getcwd()

    opt['TRIV'] = np.loadtxt(path_repo + '/data/SMPL/smpl_faces.txt', dtype='int32')

    path_evecs = path_repo + '/data/SMPL/evecs_GL_6890.npy'

    print('Loading eigen vectors...')
    opt['evecs'] = torch.from_numpy(np.load(path_evecs)).float().to(opt['device'])[:, :opt['nb_freq']]

    # datasets
    print('Loading datasets...')
    opt['dataloader_train'] = get_dataloader(opt, 'train', opt['train_batch_size'])
    opt['dataloader_vald'] = get_dataloader(opt, 'vald', opt['vald_batch_size'], shuffle=False)

    # model
    exec("from models." + opt['model_type'] + " import " + opt['model_type'])
    opt['model_class'] = eval(opt['model_type'])  # Store the class for checkpoint loading
    opt['model'] = opt['model_class'](opt).to(opt['device'])
    opt['model'].apply(init_weights)

    nb_params = sum(p.numel() for p in opt['model'].parameters() if p.requires_grad)

    print('Number of parameters:', str(nb_params), end="\n\n")

    # pytorch lightning
    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name="",
        version=str(opt['job_id']),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/' + str(opt['job_id']),
        filename='{epoch:02d}_{spatial_validation_loss:.2f}',
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
        monitor='spatial_validation_loss',
        mode='min'
    )

    progress_bar = TQDMProgressBar(
        refresh_rate=1
    )

    pl_trainer = pl.Trainer(
        accelerator='gpu', devices=1,
        profiler=profiler,
        max_epochs=opt['num_iterations'],
        check_val_every_n_epoch=opt['check_val_every_n_epoch'],
        logger=logger,
        precision=32,
        default_root_dir='checkpoints/',
        callbacks=[checkpoint_callback, progress_bar],
        deterministic=False,
        benchmark=True,
    )

    if load:
        best_checkpoint_filename = 'checkpoints/' + str(job_id) + "/last.ckpt"
        print('Loading checkpoint:', best_checkpoint_filename)
        opt['model'] = opt['model_class'].load_from_checkpoint(best_checkpoint_filename, opt=opt).to(opt["device"])

    return pl_trainer, opt


if __name__ == "__main__":
    trainer, opt = load_trainer()  # without arguments for new job

    trainer.fit(
        opt['model'],
        opt['dataloader_train'],
        opt['dataloader_vald']
    )
