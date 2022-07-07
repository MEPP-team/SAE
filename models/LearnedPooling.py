from models.Template import Template

import torch
import torch.nn as nn


class LearnedPooling(Template):
    def __init__(self, opt):
        super(LearnedPooling, self).__init__(opt)

        encoder_features = [3, 16, 32, 64, 128]

        sizes_downsample = [opt['nb_freq'], 256, 64, 32, 16]
        sizes_upsample = sizes_downsample[::-1]

        sizes_convs_encode = [opt['size_conv'], opt['size_conv'], opt['size_conv'], opt['size_conv']]
        sizes_convs_decode = sizes_convs_encode[::-1]

        encoder_linear = [encoder_features[-1] * sizes_downsample[-1], opt['size_latent']]

        decoder_linear = [opt['size_latent'], encoder_features[-1] * sizes_downsample[-1]]

        decoder_features = encoder_features[::-1]
        decoder_features[-1] = decoder_features[-2]

        self.latent_space = encoder_linear[-1]

        if opt['activation_func'] == "ReLU":
            self.activation = nn.ReLU
        elif opt['activation_func'] == "Tanh":
            self.activation = nn.Tanh
        elif opt['activation_func'] == "Sigmoid":
            self.activation = nn.Sigmoid
        elif opt['activation_func'] == "LeakyReLU":
            self.activation = nn.LeakyReLU
        elif opt['activation_func'] == "ELU":
            self.activation = nn.ELU
        else:
            print('Wrong activation')
            exit()

        # Encoder
        self.encoder_features = torch.nn.ModuleList()

        for i in range(len(encoder_features) - 1):
            self.encoder_features.append(
                torch.nn.Conv1d(
                    encoder_features[i], encoder_features[i + 1], sizes_convs_encode[i],
                    padding=sizes_convs_encode[i] // 2
                )
            )
            self.encoder_features.append(self.activation())

        self.encoder_linear = torch.nn.ModuleList()

        for i in range(len(encoder_linear) - 1):
            self.encoder_linear.append(torch.nn.Linear(encoder_linear[i], encoder_linear[i + 1]))

        # Decoder
        self.decoder_linear = torch.nn.ModuleList()

        for i in range(len(decoder_linear) - 1):
            self.decoder_linear.append(torch.nn.Linear(decoder_linear[i], decoder_linear[i + 1]))

        self.decoder_features = torch.nn.ModuleList()

        for i in range(len(decoder_features) - 1):
            self.decoder_features.append(
                torch.nn.Conv1d(
                    decoder_features[i], decoder_features[i + 1], sizes_convs_decode[i],
                    padding=sizes_convs_decode[i] // 2
                )
            )
            self.decoder_features.append(self.activation())

        self.last_conv = torch.nn.Conv1d(
            decoder_features[-1], 3, sizes_convs_decode[-1],
            padding=sizes_convs_decode[-1] // 2
        )

        # Downsampling mats
        self.downsampling_mats = torch.nn.ParameterList()

        k = 0

        for i, layer in enumerate(self.encoder_features):
            if isinstance(layer, self.activation):
                self.downsampling_mats.append(
                    torch.nn.Parameter(
                        torch.zeros(sizes_downsample[k], sizes_downsample[k+1]).to(opt["device"]),
                        requires_grad=True
                    )
                )

                k += 1

        self.upsampling_mats = torch.nn.ParameterList()

        k = 0

        for i, layer in enumerate(self.decoder_features):
            if isinstance(layer, torch.nn.Conv1d):
                self.upsampling_mats.append(
                    torch.nn.Parameter(
                        torch.zeros(sizes_upsample[k], sizes_upsample[k+1]).to(opt["device"]),
                        requires_grad=True
                    )
                )

                k += 1

        self.activation_ = self.activation()

    def enc(self, x):
        x = x.permute(0, 2, 1)

        k = 0

        for i, layer in enumerate(self.encoder_features):
            x = layer(x)

            if isinstance(layer, self.activation):
                x = torch.matmul(x, self.downsampling_mats[k])
                k += 1

        x = torch.flatten(x, start_dim=1, end_dim=2)

        for i, layer in enumerate(self.encoder_linear):
            x = layer(x)

        return x

    def dec(self, x):
        for i, layer in enumerate(self.decoder_linear):
            x = layer(x)

        x = x.view(x.shape[0], -1, self.upsampling_mats[0].shape[0])

        k = 0

        for i, layer in enumerate(self.decoder_features):
            if isinstance(layer, torch.nn.Conv1d):
                x = torch.matmul(x, self.upsampling_mats[k])
                k += 1

            x = layer(x)

        x = self.last_conv(x)

        x = x.permute(0, 2, 1)

        return x
