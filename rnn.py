"""
Define NN architecture, forward function and loss function
"""
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import custom_rnn
import json
from generic_model import generic_model


class customRNN(generic_model):

    def __init__(self, config, weights=None):
        # weights not being used currently

        super(customRNN, self).__init__(config)

        self.rnn_name = config['rnn']
        # Store important parameters
        self.feat_dim = config['n_mfcc'] + config['n_fbank']
        self.hidden_dim, self.num_phones, self.num_layers = config['hidden_dim'], config['num_phones'], config[
            'num_layers']
        self.output_dim = self.num_phones + 2  # 1 for pad and 1 for blank
        self.blank_token_id = self.num_phones + 1
        self.pad_token_id = self.num_phones

        if config['bidirectional']:
            if self.rnn_name == 'customLSTM':
                self.rnn = custom_rnn.LayerNormLSTM(self.feat_dim, self.hidden_dim, self.num_layers, True, 0.3, 0.3,
                                         bidirectional=True, layer_norm_enabled=True)
            elif self.rnn_name == 'customliGRU':
                self.rnn = custom_rnn.customliGRU(self.feat_dim, self.hidden_dim, self.num_layers, bidirectional=True)

            # In linear network, *2 for bidirectional
            self.hidden2phone = nn.Linear(self.hidden_dim * 2, self.output_dim)
        else:
            if self.rnn_name == 'customLSTM':
                self.rnn = custom_rnn.LayerNormLSTM(self.feat_dim, self.hidden_dim, self.num_layers, True, 0.2, 0.2,
                                         bidirectional=False, layer_norm_enabled=True)
            else:
                self.rnn = custom_rnn.customliGRU(self.feat_dim, self.hidden_dim, self.num_layers, bidirectional=False)

            # In linear network, *2 for bidirectional
            self.hidden2phone = nn.Linear(self.hidden_dim, self.output_dim)  # for pad token

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

    def forward(self, x, x_lens):
        """
        Forward pass through RNN
        :param x: input tensor of shape (batch size, max sequence length, feat_dim)
        :param x_lens: actual lengths of each sequence < max sequence length (since padded with zeros)
        :return: tensor of shape (batch size, max sequence length, output dim)
        """

        batch_size, seq_len, _ = x.size()
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        # now run through LSTM
        X, _ = self.rnn(X)
        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        # run through actual linear layer
        X = self.hidden2phone(X)
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size, max(x_lens), self.num_phones + 2)
        return X

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


class RNN_extra_linear(generic_model):

    def __init__(self, config, weights=None):
        # weights not being used currently

        super(RNN_extra_linear, self).__init__(config)

        self.rnn_name = config['rnn']
        # Store important parameters
        self.feat_dim = config['n_mfcc'] + config['n_fbank']
        self.hidden_dim, self.num_phones, self.num_layers = config['hidden_dim'], config['num_phones'], config[
            'num_layers']
        self.output_dim = self.num_phones + 2  # 1 for pad and 1 for blank
        self.blank_token_id = self.num_phones + 1
        self.pad_token_id = self.num_phones

        post_linear1_dim = 256
        post_linear2_dim = 128
        self.post_linear1 = nn.Linear(post_linear1_dim, post_linear2_dim)
        self.post_linear2 = nn.Linear(post_linear2_dim, self.output_dim)

        if config['bidirectional']:
            if self.rnn_name == 'LSTM':
                self.rnn = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   dropout=0.3,
                                   bidirectional=True, batch_first=True)
            else:
                self.rnn = nn.GRU(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  dropout=0.3,
                                  bidirectional=True, batch_first=True)

            # In linear network, *2 for bidirectional
            self.hidden2phone = nn.Linear(self.hidden_dim * 2, post_linear1_dim)
        else:
            if self.rnn_name == 'LSTM':
                self.rnn = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   dropout=0.3,
                                   bidirectional=False, batch_first=True)
            else:
                self.rnn = nn.GRU(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  dropout=0.3,
                                  bidirectional=True, batch_first=True)

            # In linear network, *2 for bidirectional
            self.hidden2phone = nn.Linear(self.hidden_dim, post_linear1_dim)  # for pad token

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

    def init_hidden(self):
        """
        Initialises the hidden states
        """

        hidden = next(self.parameters()).data.new(self.num_layers, self.batch_size, self.hidden_dim)
        cell = next(self.parameters()).data.new(self.num_layers, self.batch_size, self.hidden_dim)

        if self.config_file['cuda'] and torch.cuda.is_available():
            hidden = hidden.cuda()
            cell = cell.cuda()

        return hidden, cell

    def forward(self, x, x_lens):
        """
        Forward pass through RNN
        :param x: input tensor of shape (batch size, max sequence length, feat_dim)
        :param x_lens: actual lengths of each sequence < max sequence length (since padded with zeros)
        :return: tensor of shape (batch size, max sequence length, output dim)
        """

        batch_size, seq_len, _ = x.size()

        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        # now run through LSTM
        X, _ = self.rnn(X)
        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        # run through actual linear layer
        X = self.hidden2phone(X)
        X = self.post_linear1(X)
        X = self.post_linear2(X)
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size, max(x_lens), self.num_phones + 2)
        return X

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


class RNN(generic_model):

    def __init__(self, config, weights=None):
        # weights not being used currently

        super(RNN, self).__init__(config)

        self.rnn_name = config['rnn']
        # Store important parameters
        self.feat_dim = config['n_mfcc'] + config['n_fbank']
        self.hidden_dim, self.num_phones, self.num_layers = config['hidden_dim'], config['num_phones'], config[
            'num_layers']
        self.output_dim = self.num_phones + 2  # 1 for pad and 1 for blank
        self.blank_token_id = self.num_phones + 1
        self.pad_token_id = self.num_phones

        if config['bidirectional']:
            if self.rnn_name == 'LSTM':
                self.rnn = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   dropout=0.3,
                                   bidirectional=True, batch_first=True)
            else:
                self.rnn = nn.GRU(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  dropout=0.3,
                                  bidirectional=True, batch_first=True)

            # In linear network, *2 for bidirectional
            self.hidden2phone = nn.Linear(self.hidden_dim * 2, self.output_dim)
        else:
            if self.rnn_name == 'LSTM':
                self.rnn = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   dropout=0.3,
                                   bidirectional=False, batch_first=True)
            else:
                self.rnn = nn.GRU(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  dropout=0.3,
                                  bidirectional=True, batch_first=True)

            # In linear network, *2 for bidirectional
            self.hidden2phone = nn.Linear(self.hidden_dim, self.output_dim)  # for pad token

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

    def forward(self, x, x_lens):
        """
        Forward pass through RNN
        :param x: input tensor of shape (batch size, max sequence length, feat_dim)
        :param x_lens: actual lengths of each sequence < max sequence length (since padded with zeros)
        :return: tensor of shape (batch size, max sequence length, output dim)
        """

        batch_size, seq_len, _ = x.size()
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        # now run through LSTM
        X, _ = self.rnn(X)
        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        # run through actual linear layer
        X = self.hidden2phone(X)
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size, max(x_lens), self.num_phones + 2)
        return X

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
