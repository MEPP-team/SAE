import time
import pytorch_lightning as pl
import torch
import numpy as np

from models.utils.error_meshes import error_meshes_mm


class Template(pl.LightningModule):
    def __init__(self, opt):
        super(Template, self).__init__()
        self.opt = opt
        self.needed_freq = opt['nb_freq']
        self.test_step_outputs = []

    def forward(self, x):
        latent = self.enc(x)
        output = self.dec(latent)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.opt['learning_rate'],
            weight_decay=0
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.99,
            patience=3
        )

        # reduce every epoch (default)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'spatial_validation_loss',
            'strict': False
        }

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        _, inputs = train_batch
        inputs = inputs[:, :self.needed_freq]
        outputs = self.forward(inputs)

        loss = self.opt['loss'](outputs, inputs)
        self.log("spectral_train_loss", loss, on_step=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        _, inputs, vertices = val_batch

        inputs = inputs[:, :self.needed_freq]

        outputs = self.forward(inputs)

        spectral_loss = self.opt['loss'](outputs, inputs)
        self.log("spectral_validation_loss", spectral_loss)

        outputs = torch.matmul(self.opt['evecs'], outputs)

        losses = torch.mean(error_meshes_mm(vertices, outputs), dim=1)
        spatial_loss = torch.mean(losses) * 1e3

        self.log("spatial_validation_loss", spatial_loss)
        return {"spectral_loss": spectral_loss, "spatial_loss": spatial_loss}

    def test_step(self, batch, batch_idx):
        _, inputs, vertices = batch

        inputs = inputs[:, :self.needed_freq]

        outputs = self.forward(inputs)

        outputs = torch.matmul(self.opt['evecs'], outputs)

        losses = torch.mean(error_meshes_mm(vertices, outputs), dim=1)
        losses = losses.tolist()
        self.test_step_outputs.append(losses)
        return losses

    def on_test_epoch_end(self):
        # output is list of list of different sizes (smaller last batch), so we have to work on lists
        flatten_list = [element for sublist in self.test_step_outputs for element in sublist]
        flatten_array = np.array(flatten_list)*1e3

        self.log("mean spatial_loss", flatten_array.mean())
        self.log("median spatial_loss", np.median(flatten_array))
        self.log("std spatial_loss", flatten_array.std())
        self.log("min spatial_loss", flatten_array.min())
        self.log("max spatial_loss", flatten_array.max())
        
        self.test_step_outputs.clear()

    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_epoch_end(self):
        self.log("epoch_time", time.time() - self.start_time)
