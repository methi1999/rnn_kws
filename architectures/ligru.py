"""
Define NN architecture, forward function and loss function
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm

import utils
from architectures.generic_model import generic_model


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :, getattr(torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda])().long(), :
        ]
    return x.view(xsize)


class liGRU(generic_model):
    def __init__(self, config, to_do):

        super(liGRU, self).__init__(config)

        self.phone_to_id = utils.load_phone_mapping(config)
        self.rnn_name = config['rnn']
        # Store important parameters
        self.input_dim = config['n_mfcc'] + config['n_fbank']
        self.hidden_dim, self.num_phones, self.num_layers = config['hidden_dim'], config['num_phones'], config[
            'num_layers']
        self.output_dim = self.num_phones + 2  # 1 for pad and 1 for blank
        self.blank_token_id = self.phone_to_id['BLANK']

        self.ligru_act = nn.LeakyReLU(0.2)
        self.is_bidirectional = config['bidirectional']
        self.ligru_orthinit = True
        self.dropout_vals = [0.2] * self.num_layers
        self.hidden_dim_layers = [self.hidden_dim] * self.num_layers
        self.use_cuda = torch.cuda.is_available() and config['use_cuda']

        if to_do == 'train':
            self.train_flag = True
        else:
            self.train_flag = False

        self.use_batchnorm = True
        self.use_batchnorm_inp = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.act = nn.ModuleList([])  # Activations

        # Batch Norm
        if self.use_batchnorm:
            self.bn_wh = nn.ModuleList([])
            self.bn_wz = nn.ModuleList([])

        # Input batch normalization
        if self.use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        current_input = self.input_dim

        # Initialization of hidden layers
        for i in range(self.num_layers):

            # Activations
            self.act.append(self.ligru_act)

            add_bias = True

            if self.use_batchnorm:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.hidden_dim_layers[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.hidden_dim_layers[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.hidden_dim_layers[i], self.hidden_dim_layers[i], bias=False))
            self.uz.append(nn.Linear(self.hidden_dim_layers[i], self.hidden_dim_layers[i], bias=False))

            if self.ligru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)

            if self.use_batchnorm:
                # batch norm initialization
                self.bn_wh.append(nn.BatchNorm1d(self.hidden_dim_layers[i], momentum=0.05))
                self.bn_wz.append(nn.BatchNorm1d(self.hidden_dim_layers[i], momentum=0.05))

            if self.is_bidirectional:
                current_input = 2 * self.hidden_dim_layers[i]
            else:
                current_input = self.hidden_dim_layers[i]

        self.out_dim = self.hidden_dim_layers[-1] + self.is_bidirectional * self.hidden_dim_layers[-1]

        if self.is_bidirectional:
            self.hidden_2_phone = nn.Linear(2 * self.hidden_dim, self.output_dim)
        else:
            self.hidden_2_phone = nn.Linear(self.hidden_dim, self.output_dim)

        self.loss_func = torch.nn.CTCLoss(blank=self.blank_token_id, reduction='mean', zero_infinity=False)
        print("Using CTC loss")

        optimizer = config['train']['optim']
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=config['train']['lr'], momentum=0.9)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=config['train']['lr'])

    def forward(self, x_batch, lens):

        final = torch.zeros((x_batch.shape[0], x_batch.shape[1], self.output_dim))

        for eg, x in enumerate(x_batch):

            x = x_batch[eg:eg+1, :lens[eg], :]

            if self.use_batchnorm_inp:
                x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
                x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

            for i in range(self.num_layers):

                # Initial state and concatenation
                if self.is_bidirectional:
                    h_init = torch.zeros(2 * x.shape[1], self.hidden_dim_layers[i])
                    x = torch.cat([x, flip(x, 0)], 1)
                else:
                    h_init = torch.zeros(x.shape[1], self.hidden_dim_layers[i])

                # Drop mask initialization (same mask for all time steps)
                if self.train_flag:
                    drop_mask = torch.bernoulli(
                        torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.dropout_vals[i])
                    )
                else:
                    drop_mask = torch.FloatTensor([1 - self.dropout_vals[i]])

                if self.use_cuda:
                    h_init = h_init.cuda()
                    drop_mask = drop_mask.cuda()

                # Feed-forward affine transformations (all steps in parallel)
                wh_out = self.wh[i](x)
                wz_out = self.wz[i](x)

                # Apply batch norm if needed (all steos in parallel)
                if self.use_batchnorm:
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

                    hiddens.append(ht)

                # Stacking hidden states
                h = torch.stack(hiddens)

                # Bidirectional concatenations
                if self.is_bidirectional:
                    h_f = h[:, 0: int(x.shape[1] / 2)]
                    h_b = flip(h[:, int(x.shape[1] / 2): x.shape[1]].contiguous(), 0)
                    h = torch.cat([h_f, h_b], 2)

                # Setup x for the next hidden layer
                x = h

            final[eg, :lens[eg], :] = self.hidden_2_phone(x)

        return final

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