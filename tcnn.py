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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(generic_model):
    def __init__(self, config, weights=None):
        super(TCN, self).__init__(config)

        self.rnn_name = config['rnn']
        self.feat_dim = config['n_mfcc'] + config['n_fbank']
        self.hidden_dim, self.num_phones, self.num_layers = config['hidden_dim'], config['num_phones'], config[
            'num_layers']
        self.output_dim = self.num_phones + 2  # 1 for pad and 1 for blank
        self.blank_token_id = self.num_phones + 1
        self.pad_token_id = self.num_phones
        self.num_channels = [self.hidden_dim] * (self.num_layers - 1) + [self.output_dim]
        kernel_size, dropout = 3, 0.2

        layers = []
        num_levels = len(self.num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.feat_dim if i == 0 else self.num_channels[i - 1]
            out_channels = self.num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

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
            fname = config['dir']['dataset'] + 'phone_mapping.json'
            with open(fname, 'r') as f:
                self.phone_to_id = json.load(f)

            self.weights = np.array([x[1] for x in self.phone_to_id.values()])

            assert len(self.phone_to_id) == config['num_phones'] + 1  # 1 for pad token

        except:
            print("Can't find phone mapping")
            exit(0)

    def forward(self, x, lens):
        x = x.transpose(2, 1)
        # print(x.shape)
        return self.network(x).transpose(1, 2)

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
        # print(torch.sum(outputs, dim=2))
        return self.loss_func(outputs, labels, input_lens, label_lens)


class bidirectional_TCN(generic_model):
    def __init__(self, config, weights=None):
        super(bidirectional_TCN, self).__init__(config)

        self.rnn_name = config['rnn']
        self.feat_dim = config['n_mfcc'] + config['n_fbank']
        self.hidden_dim, self.num_phones, self.num_layers = config['hidden_dim'], config['num_phones'], config[
            'num_layers']
        self.output_dim = self.num_phones + 2  # 1 for pad and 1 for blank
        self.blank_token_id = self.num_phones + 1
        self.pad_token_id = self.num_phones
        self.num_channels = [self.hidden_dim] * self.num_layers
        kernel_size, dropout = 3, 0.2

        self.final_conv = weight_norm(nn.Conv1d(2 * self.hidden_dim, self.output_dim, kernel_size,
                                                stride=1, padding=(kernel_size - 1)//2,
                                                dilation=1))

        forward_layers = []
        backward_layers = []
        num_levels = len(self.num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.feat_dim if i == 0 else self.num_channels[i - 1]
            out_channels = self.num_channels[i]
            forward_layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                             padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            backward_layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                              padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.forward_network = nn.Sequential(*forward_layers)
        self.backward_network = nn.Sequential(*backward_layers)

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
            fname = config['dir']['dataset'] + 'phone_mapping.json'
            with open(fname, 'r') as f:
                self.phone_to_id = json.load(f)

            self.weights = np.array([x[1] for x in self.phone_to_id.values()])

            assert len(self.phone_to_id) == config['num_phones'] + 1  # 1 for pad token

        except:
            print("Can't find phone mapping")
            exit(0)

    def forward(self, x, lens):
        # x is of size (batch-size x time x feat_dim)
        # pass output is of size (batch-size x time x 41 (phones))
        forward_pass = self.forward_network(x.transpose(2, 1))
        # flip along time axis
        x = torch.flip(x, [1])
        backward_pass = self.backward_network(x.transpose(2, 1))
        flipped_backward = torch.flip(backward_pass, [1])
        # concatenate past and future information
        concat = torch.cat((forward_pass, flipped_backward), 1)
        return self.final_conv(concat).transpose(1, 2)

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
        # print(torch.sum(outputs, dim=2))
        return self.loss_func(outputs, labels, input_lens, label_lens)