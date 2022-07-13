import argparse
import os
import webdataset as wds
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

        try:
            os.mkdir('checkpoints/' + str(opt['job_id']))
        except:
            print('Folder for job ' + str(opt['job_id']) + ' already exists.')
            exit()

        with open('checkpoints/' + str(opt['job_id']) + '/infos.json', "w") as outfile:
            json.dump(opt, outfile, sort_keys=True, indent=4)

    print('Options:\n', json.dumps(opt, sort_keys=True, indent=4), end="\n\n")

    return opt


def get_dataloader(opt, dataset_type, batch_size, wanted_index=None, shuffle=True):
    dataloader_slice = wanted_index is not None

    url = opt['path'] + "/" + dataset_type + ".tar"

    if dataset_type == 'train':
        tuple_type = ("infos.json", "coeffs.pth")

        if dataloader_slice:
            dataset = wds.WebDataset(url).slice(wanted_index, opt['nb_train']).decode().to_tuple(*tuple_type)
        else:
            if shuffle:
                dataset = wds.WebDataset(url).shuffle(opt['nb_train'], initial=opt['nb_train']).decode().to_tuple(*tuple_type)
            else:
                dataset = wds.WebDataset(url).decode().to_tuple(*tuple_type)
    else:
        tuple_type = ("infos.json", "coeffs.pth", "vertices.pth")
        if dataloader_slice:
            dataset = wds.WebDataset(url).slice(wanted_index, opt['nb_evals']).decode().to_tuple(*tuple_type)
        else:
            dataset = wds.WebDataset(url).decode().to_tuple(*tuple_type)

    return wds.WebLoader(
        dataset,
        num_workers=opt["num_workers"],
        batch_size=batch_size,
        pin_memory=True,
    )


def load_trainer(job_id=None, profiler="simple"):
    if job_id is None:
        load = False
    else:
        load = True

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
    opt['TRIV'] = np.loadtxt(opt['path'] + '/../mesh.triv', dtype='int32') - 1

    path_evecs = opt['path'] + '/../mesh.evecs_6890'

    print('Loading eigen vectors...', end='')
    opt['evecs'] = torch.from_numpy(np.load(path_evecs)).float().to(opt['device'])[:, :opt['nb_freq']]
    print(' done.')

    # datasets
    opt['dataloader_train'] = get_dataloader(opt, 'train', opt['train_batch_size'])
    opt['dataloader_test'] = get_dataloader(opt, 'test', opt['test_batch_size'])

    # model
    exec("from models." + opt['model_type'] + " import " + opt['model_type'])
    opt['model'] = eval(opt['model_type'])(opt).to(opt['device'])
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
        every_n_epochs=opt['check_val_every_n_epoch'],
        save_top_k=0,
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
        opt['model'] = opt['model'].load_from_checkpoint(best_checkpoint_filename, opt=opt).to(opt["device"])

    return pl_trainer, opt


if __name__ == "__main__":
    trainer, opt = load_trainer()  # without arguments for new job

    trainer.fit(
        opt['model'],
        opt['dataloader_train'],
        opt['dataloader_test']
    )
