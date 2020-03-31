import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def no_dropout(x): return x


def no_layer_norm(x): return x


def get_indicator(lengths, max_length=None):
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.Tensor(lengths)
    lengths_size = lengths.size()

    flat_lengths = lengths.view(-1, 1)

    if not max_length:
        max_length = lengths.max()
    unit_range = torch.arange(max_length)
    # flat_range = torch.stack([unit_range] * flat_lengths.size()[0],
    #                          dim=0)
    flat_range = unit_range.repeat(flat_lengths.size()[0], 1)
    flat_indicator = flat_range < flat_lengths

    return flat_indicator.view(lengths_size + (-1, 1))


def create_lstm_init_state(hidden_size, learn_init_state):
    init_hidden = nn.Parameter(torch.zeros(hidden_size), learn_init_state)
    init_cell = nn.Parameter(torch.zeros(hidden_size), learn_init_state)

    init_state = (init_hidden, init_cell)
    _init_state = nn.ParameterList(init_state)

    return init_state, _init_state


def enable_cuda(model, arg):
    if is_cuda_enabled(model):
        arg = arg.cuda()
    else:
        arg = arg.cpu()
    return arg


def is_cuda_enabled(model):
    return next(model.parameters()).is_cuda


class LSTMFrame(nn.Module):
    def __init__(self, rnn_cells, dropout=0, batch_first=False, bidirectional=False):
        """
        :param rnn_cells: ex) [(cell_0_f, cell_0_b), (cell_1_f, cell_1_b), ..]
        :param dropout:
        :param bidirectional:
        """
        super().__init__()

        if bidirectional:
            assert all(len(pair) == 2 for pair in rnn_cells)
        elif not any(isinstance(rnn_cells[0], iterable)
                     for iterable in [list, tuple, nn.ModuleList]):
            rnn_cells = tuple((cell,) for cell in rnn_cells)

        self.rnn_cells = nn.ModuleList(nn.ModuleList(pair)
                                       for pair in rnn_cells)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = len(rnn_cells)

        if dropout > 0 and self.num_layers > 1:
            # dropout is applied to output of each layer except the last layer
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = no_dropout

    def align_sequence(self, seq, lengths, shift_right):
        """
        :param seq: (seq_len, batch_size, *)
        """
        multiplier = 1 if shift_right else -1
        example_seqs = torch.split(seq, 1, dim=1)
        max_length = max(lengths)
        shifted_seqs = []
        for i in range(len(lengths)):
            example_seq, length = example_seqs[i], lengths[i]
            shift = ((max_length - length) * multiplier).item()
            shifted_seqs.append(example_seq.roll(shifts=shift, dims=0))
        return torch.cat(shifted_seqs, dim=1)

    def forward(self, input, init_state=None):
        """
        :param input: a tensor(s) of shape (seq_len, batch, input_size)
        :param init_state: (h_0, c_0) where the size of both is (num_layers * num_directions, batch, hidden_size)
        :returns:
        - output: (seq_len, batch, num_directions * hidden_size)
        - h_n: (num_layers * num_directions, batch, hidden_size)
        - c_n: (num_layers * num_directions, batch, hidden_size)
        """

        if isinstance(input, torch.nn.utils.rnn.PackedSequence):
            input_packed = True
            input, lengths = pad_packed_sequence(input)
            if max(lengths) == min(lengths):
                uniform_length = True
            else:
                uniform_length = False
            assert max(lengths) == input.size()[0]
        else:
            input_packed = False
            lengths = [input.size()[0]] * input.size()[1]
            uniform_length = True

        if not uniform_length:
            indicator = get_indicator(lengths)
            # valid_example_nums = indicator.sum(0)

        if init_state is None:
            # init_state with heterogenous hidden_size
            init_hidden = init_cell = [
                enable_cuda(self, torch.zeros(
                    input.size()[1], self.rnn_cells[layer_idx][direction].hidden_size))
                for layer_idx in range(self.num_layers)
                for direction in range(self.num_directions)]
            init_state = init_hidden, init_cell

        init_hidden, init_cell = init_state

        last_hidden_list = []
        last_cell_list = []

        layer_output = input

        for layer_idx in range(self.num_layers):
            layer_input = layer_output
            if layer_idx != 0:
                layer_input = self.dropout(layer_input)

            direction_output_list = []

            for direction in range(self.num_directions):
                cell = self.rnn_cells[layer_idx][direction]
                state_idx = layer_idx * self.num_directions + direction
                step_state = (init_hidden[state_idx], init_cell[state_idx])

                direction_output = enable_cuda(self, torch.zeros(
                    layer_input.size()[:2] + (cell.hidden_size,)))  # (seq_len, batch_size, hidden_size)
                step_state_list = []

                if direction == 0:
                    step_input_gen = enumerate(layer_input)
                else:
                    step_input_gen = reversed(list(enumerate(
                        layer_input if uniform_length else
                        self.align_sequence(layer_input, lengths, True))))

                for seq_idx, cell_input in step_input_gen:
                    # if not uniform_length:  # for speed enhancement
                    #     cell_input = cell_input[:valid_example_nums[seq_idx]]
                    #     step_state = step_state[:valid_example_nums[seq_idx]]
                    h, c = step_state = cell(cell_input, step_state)
                    # if uniform_length:
                    direction_output[seq_idx] = h
                    step_state_list.append(step_state)
                    # else:       # for speed enhancement
                    #     direction_output[seq_idx][? :?] = h
                    #     step_state_list.append(step_state)
                if direction == 1 and not uniform_length:
                    direction_output = self.align_sequence(
                        direction_output, lengths, False)

                if uniform_length:
                    # hidden & cell's size = (batch, hidden_size)
                    direction_last_hidden, direction_last_cell = step_state_list[-1]
                else:
                    direction_last_hidden, direction_last_cell = map(
                        lambda x: torch.stack([x[length - 1][example_id]
                                               for example_id, length in enumerate(lengths)], dim=0),
                        zip(*step_state_list))

                direction_output_list.append(direction_output)
                last_hidden_list.append(direction_last_hidden)
                last_cell_list.append(direction_last_cell)

            if self.num_directions == 2:
                layer_output = torch.stack(direction_output_list, dim=2).view(
                    direction_output_list[0].size()[:2] + (-1,))
            else:
                layer_output = direction_output_list[0]

        output = layer_output
        last_hidden_tensor = torch.stack(last_hidden_list, dim=0)
        last_cell_tensor = torch.stack(last_cell_list, dim=0)

        if not uniform_length:
            # the below one line code cleans out trash values beyond the range of lengths.
            # actually, the code is for debugging, so it can be removed to enhance computing speed slightly.
            output = (output.transpose(0, 1) * enable_cuda(self, indicator).float()).transpose(0, 1)

        if input_packed:
            # always batch_first=False --> trick to process input regardless of batch_first option
            output = pack_padded_sequence(output, lengths)

        return output, (last_hidden_tensor, last_cell_tensor)


class LSTMCell(nn.Module):
    """
    standard LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fiou_linear = nn.Linear(input_size + hidden_size, hidden_size * 4)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    def forward(self, input, state):
        """
        :param input: a tensor of of shape (batch_size, input_size)
        :param state: a pair of a hidden tensor and a cell tensor whose shape is (batch_size, hidden_size).
                      ex. (h_0, c_0)
        :returns: 1-dimensional hidden and cell
        """
        hidden_tensor, cell_tensor = state

        fio_linear, u_linear = torch.split(
            self.fiou_linear(torch.cat([input, hidden_tensor], dim=1)),
            self.hidden_size * 3, dim=1)

        f, i, o = torch.split(torch.sigmoid(fio_linear),
                              self.hidden_size, dim=1)
        u = torch.tanh(u_linear)

        new_cell = i * u + (f * cell_tensor)
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell


class LayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, layer_norm_enabled=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, hidden_size, bias=not layer_norm_enabled)

        # if dropout is not None:
        #     if isinstance(dropout, nn.Dropout):
        #         self.dropout = dropout
        #     elif dropout > 0:
        #         self.dropout = nn.Dropout(dropout)
        #     else:
        #         self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = no_layer_norm

    def forward(self, input, hidden):
        """
        :param input: a tensor of of shape (batch_size, input_size)
        :param state: a hidden tensor of shape (batch_size, hidden_size).
                      ex. (h_0, c_0)
        :returns: hidden and cell
        """
        return torch.tanh(self.layer_norm(self.linear(
            torch.cat([input, hidden], dim=1))))


class LayerNormLSTMCell(nn.Module):
    """
    It's based on tf.contrib.rnn.LayerNormBasicLSTMCell
    Reference:
    - https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LayerNormBasicLSTMCell
    - https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1335
    """

    def __init__(self, input_size, hidden_size, dropout=None, layer_norm_enabled=True, cell_ln=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fiou_linear = nn.Linear(
            input_size + hidden_size, hidden_size * 4, bias=not layer_norm_enabled)

        if dropout is not None:
            # recurrent dropout is applied
            if isinstance(dropout, nn.Dropout):
                self.dropout = dropout
            elif dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.fiou_ln_layers = nn.ModuleList(
                nn.LayerNorm(hidden_size) for _ in range(4))
            # self.fiou_ln_layers = nn.ModuleList(
            #     nn.LayerNorm(hidden_size) for _ in range(3))
            # self.fiou_ln_layers.append(
            #     nn.LayerNorm(hidden_size) if u_ln is None else u_ln)
            self.cell_ln = nn.LayerNorm(
                hidden_size) if cell_ln is None else cell_ln
        else:
            assert cell_ln is None
            # assert u_ln is cell_ln is None
            self.fiou_ln_layers = (no_layer_norm,) * 4
            self.cell_ln = no_layer_norm
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    def forward(self, input, state):
        """
        :param input: a tensor of of shape (batch_size, input_size)
        :param state: a pair of a hidden tensor and a cell tensor whose shape is (batch_size, hidden_size).
                      ex. (h_0, c_0)
        :returns: hidden and cell
        """
        hidden_tensor, cell_tensor = state

        fiou_linear = self.fiou_linear(
            torch.cat([input, hidden_tensor], dim=1))
        fiou_linear_tensors = fiou_linear.split(self.hidden_size, dim=1)

        # if self.layer_norm_enabled:
        fiou_linear_tensors = tuple(ln(tensor) for ln, tensor in zip(
            self.fiou_ln_layers, fiou_linear_tensors))

        f, i, o = tuple(torch.sigmoid(tensor)
                        for tensor in fiou_linear_tensors[:3])
        u = self.dropout(torch.tanh(fiou_linear_tensors[3]))

        new_cell = self.cell_ln(i * u + (f * cell_tensor))
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell


class LayerNormLSTM(LSTMFrame):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, r_dropout=0, bidirectional=False, layer_norm_enabled=True):
        r_dropout_layer = nn.Dropout(r_dropout)
        rnn_cells = tuple(
            tuple(
                LayerNormLSTMCell(
                    input_size if layer_idx == 0 else hidden_size * (2 if bidirectional else 1),
                    hidden_size,
                    dropout=r_dropout_layer,
                    layer_norm_enabled=layer_norm_enabled)
                for _ in range(2 if bidirectional else 1))
            for layer_idx in range(num_layers))

        super().__init__(rnn_cells=rnn_cells, dropout=dropout,
                         batch_first=batch_first, bidirectional=bidirectional)
