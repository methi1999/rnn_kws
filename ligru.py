"""
Define NN architecture, forward function and loss function
"""
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
import json
from generic_model import generic_model


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :, getattr(torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda])().long(), :
        ]
    return x.view(xsize)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class liGRU(generic_model):
    def __init__(self, config, to_do):

        super(liGRU, self).__init__(config)

        self.rnn_name = config['rnn']
        # Store important parameters
        self.input_dim = config['n_mfcc'] + config['n_fbank']
        self.hidden_dim, self.num_phones, self.num_layers = config['hidden_dim'], config['num_phones'], config[
            'num_layers']
        self.output_dim = self.num_phones + 2  # 1 for pad and 1 for blank
        self.blank_token_id = self.num_phones + 1
        self.pad_token_id = self.num_phones

        self.ligru_act = nn.LeakyReLU(0.2)
        self.bidir = config['bidirectional']
        self.ligru_orthinit = True
        self.ligru_drop = [0.2] * self.num_layers
        self.ligru_lay = [self.hidden_dim] * self.num_layers
        self.use_cuda = torch.cuda.is_available() and config['cuda']

        if to_do == 'train':
            self.test_flag = False
        else:
            self.test_flag = True

        self.ligru_use_batchnorm = False
        self.ligru_use_laynorm = False
        self.ligru_use_laynorm_inp = True
        self.ligru_use_batchnorm_inp = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.ligru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.ligru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_ligru_lay = len(self.ligru_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_ligru_lay):

            # Activations
            self.act.append(self.ligru_act)

            add_bias = True

            if self.ligru_use_laynorm or self.ligru_use_batchnorm:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))

            if self.ligru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.ligru_lay[i]))

            if self.bidir:
                current_input = 2 * self.ligru_lay[i]
            else:
                current_input = self.ligru_lay[i]

        self.out_dim = self.ligru_lay[i] + self.bidir * self.ligru_lay[i]

        if self.bidir:
            self.hidden_2_phone = nn.Linear(2 * self.hidden_dim, self.output_dim)
        else:
            self.hidden_2_phone = nn.Linear(self.hidden_dim, self.output_dim)

        loss, optimizer = config['train']['loss_func'], config['train']['optim']
        loss_found, optim_found = False, False

        self.loss_func = torch.nn.CTCLoss(blank=self.blank_token_id, reduction='mean', zero_infinity=False)
        print("Using CTC loss")
        loss_found = True

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=config['train']['lr'], momentum=0.9)
            optim_found = True
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=config['train']['lr'])
            optim_found = True

        if loss_found is False or optim_found is False:
            print("Can't find desired loss function/optimizer")
            exit(0)

        # Load mapping of phone to id
        try:
            fname = config['dir']['dataset'] + 'lstm_mapping.json'
            with open(fname, 'r') as f:
                self.phone_to_id = json.load(f)

            self.weights = np.array([x[1] for x in self.phone_to_id.values()])

            assert len(self.phone_to_id) == config['num_phones'] + 1  # 1 for pad token

        except:
            print("Can't find phone mapping")
            exit(0)

    def forward(self, x, lens):

        # Applying Layer/Batch Norm
        if bool(self.ligru_use_laynorm_inp):
            x = self.ln0(x)

        if bool(self.ligru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_ligru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.ligru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.ligru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == 'train':
                drop_mask = torch.bernoulli(
                    torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.ligru_drop[i])
                )
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.ligru_use_batchnorm:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # ligru equation
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                at = wh_out[k] + self.uh[i](ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand

                if self.ligru_use_laynorm:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0: int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2): x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return self.hidden_2_phone(x)

    def calculate_loss(self, outputs, labels, input_lens, label_lens):
        """
        Reshapes tensors as required and pass to CTC function
        :param outputs: tensor of shape (batch size, max sequence length, output dim) from forward pass
        :param labels: tensor of shape (batch size, output dim). Integer denoting the class
        :param input_lens: lengths of input sequences
        :param label_lens: lengths of ground truth phones
        :return: CTC loss
        """

        outputs = nn.functional.log_softmax(outputs.transpose(0, 1), dim=2)

        return self.loss_func(outputs, labels, input_lens, label_lens)