"""
The main driver file responsible for training, testing and extracting features
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from copy import deepcopy
import pickle
from read_yaml import read_yaml
from dataloader import timit_loader
import utils
from beam_search import decode


class dl_model():

    def __init__(self, mode):

        # Read config fielewhich contains parameters
        self.config = read_yaml()
        self.mode = mode

        if self.config['rnn'] == 'liGRU':
            from model import liGRU as Model
        elif self.config['rnn'] == 'GRU' or self.config['rnn'] == 'LSTM':
            from model import RNN as Model
        else:
            print("Model import failed")
            exit(0)
        # Architecture name decides prefix for storing models and plots
        feature_dim = self.config['n_fbank'] + self.config['n_mfcc']
        self.arch_name = '_'.join(
            [self.config['rnn'], str(self.config['num_layers']), str(self.config['hidden_dim']), str(feature_dim)])

        print("Architecture:", self.arch_name)
        # Change paths for storing models
        self.config['dir']['models'] = self.config['dir']['models'].split('/')[0] + '_' + self.arch_name + '/'
        self.config['dir']['plots'] = self.config['dir']['plots'].split('/')[0] + '_' + self.arch_name + '/'

        # Make folders if DNE
        if not os.path.exists(self.config['dir']['models']):
            os.mkdir(self.config['dir']['models'])
        if not os.path.exists(self.config['dir']['plots']):
            os.mkdir(self.config['dir']['plots'])
        if not os.path.exists(self.config['dir']['pickle']):
            os.mkdir(self.config['dir']['pickle'])

        self.cuda = (self.config['cuda'] and torch.cuda.is_available())

        # load/initialise metrics to be stored and load model
        if mode == 'train' or mode == 'test':

            self.plots_dir = self.config['dir']['plots']
            # store hyperparameters
            self.total_epochs = self.config['train']['epochs']
            self.test_every = self.config['train']['test_every_epoch']
            self.test_per = self.config['train']['test_per_epoch']
            self.print_per = self.config['train']['print_per_epoch']
            self.save_every = self.config['train']['save_every']
            self.plot_every = self.config['train']['plot_every']

            # declare model

            # dataloader which returns batches of data
            self.train_loader = timit_loader('train', self.config)
            self.test_loader = timit_loader('test', self.config)
            self.model = Model(self.config, mode)

            self.start_epoch = 1
            self.edit_dist = []
            self.train_losses, self.test_losses = [], []

        else:

            self.model = Model(self.config, mode)

        if self.cuda:
            self.model.cuda()

        # resume training from some stored model
        if self.mode == 'train' and self.config['train']['resume']:
            self.start_epoch, self.train_losses, self.test_losses, self.edit_dist = self.model.load_model(mode,
                                                                                                          self.model.rnn_name,
                                                                                                          self.model.num_layers,
                                                                                                          self.model.hidden_dim)
            self.start_epoch += 1

        # load best model for testing/feature extraction
        elif self.mode == 'test' or mode == 'test_one':
            self.model.load_model(mode, self.config['rnn'], self.model.num_layers, self.model.hidden_dim)

        # Replacement phones
        self.replacement = utils.replacement_dict()

    # Train the model
    def train(self):

        print("Starting training at t =", datetime.datetime.now())
        print('Batches per epoch:', len(self.train_loader))
        self.model.train()

        # when to print losses during the epoch
        print_range = list(np.linspace(0, len(self.train_loader), self.print_per + 2, dtype=np.uint32)[1:-1])
        if self.test_per == 0:
            test_range = []
        else:
            test_range = list(np.linspace(0, len(self.train_loader), self.test_per + 2, dtype=np.uint32)[1:-1])

        for epoch in range(self.start_epoch, self.total_epochs + 1):

            print("Epoch:", str(epoch))
            epoch_loss = 0.0
            # i used for monitoring batch and printing loss, etc.
            i = 0

            while True:

                i += 1

                # Get batch of feature vectors, labels and lengths along with status (when to end epoch)
                inputs, labels, input_lens, label_lens, status = self.train_loader.return_batch()
                # print(input_lens, label_lens)
                inputs, labels, = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(np.array(labels)).long()
                input_lens, label_lens = torch.from_numpy(np.array(input_lens)).long(), torch.from_numpy(
                    np.array(label_lens)).long()

                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    input_lens = input_lens.cuda()
                    label_lens = label_lens.cuda()

                # zero the parameter gradients
                self.model.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs, input_lens)
                loss = self.model.calculate_loss(outputs, labels, input_lens, label_lens)
                loss.backward()

                # clip gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.model.optimizer.step()

                # store loss
                epoch_loss += loss.item()

                # print loss
                if i in print_range:
                    try:
                        print('After %i batches, Current Loss = %.7f, Avg. Loss = %.7f' % (
                            i + 1, epoch_loss / (i + 1), np.mean(np.array([x[0] for x in self.train_losses]))))
                    except:
                        pass

                # test model periodically
                if i in test_range:
                    self.test(epoch)
                    self.model.train()

                # Reached end of dataset
                if status == 1:
                    break

            # Store tuple of training loss and epoch number
            self.train_losses.append((epoch_loss / len(self.train_loader), epoch))

            # test every 5 epochs in the beginning and then every fixed no of epochs specified in config file
            # useful to see how loss stabilises in the beginning
            # save model
            if epoch % self.save_every == 0:
                self.model.save_model(False, epoch, self.train_losses, self.test_losses, self.edit_dist,
                                      self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)

            if epoch % 5 == 0 and epoch < self.test_every:
                self.test(epoch)
                self.model.train()
            elif epoch % self.test_every == 0:
                self.test(epoch)
                self.model.train()
            # plot loss and accuracy
            if epoch % self.plot_every == 0:
                self.plot_loss_acc(epoch)

    # test model
    def test(self, epoch=None):

        self.model.eval()
        # edit distance of batch
        edit_dist_batch = 0
        # number of sequences
        total_seq = 0
        # decode type
        decode_type = self.config['decode_type']
        # operations dictionary for calculating probabilities
        num_ph = self.model.num_phones
        op_dict = {}
        for i in range(num_ph):
            op_dict[i] = {'matches': 0, 'insertions': 0, 'deletions': 0,
                          'substitutions': np.zeros(self.model.num_phones), 'total': 0}

        print("Testing...")
        print('Total batches:', len(self.test_loader))
        test_loss = 0

        k = 0

        with torch.no_grad():

            while True:

                # retrieve batch from dataloader
                inputs, labels, input_lens, label_lens, status = self.test_loader.return_batch()
                # convert to tensors
                inputs, labels, input_lens, label_lens = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(
                    np.array(labels)).long(), torch.from_numpy(np.array(input_lens)).long(), torch.from_numpy(
                    np.array(label_lens)).long()

                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    input_lens = input_lens.cuda()
                    label_lens = label_lens.cuda()

                # zero the parameter gradients
                self.model.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs, input_lens)
                loss = self.model.calculate_loss(outputs, labels, input_lens, label_lens)
                test_loss += loss.item()

                outputs = outputs.cpu().numpy()
                labels = labels.cpu().numpy()

                k += outputs.shape[0]

                # calculated edit distance between ground truth and predicted sequence
                for i in range(outputs.shape[0]):
                    # predict by argmax
                    if decode_type == 'max':
                        # argmax over the phone channel
                        argmaxed = np.argmax(outputs, axis=2)
                        seq = list(argmaxed[i][:input_lens[i]])
                        # collapse neighbouring and remove blank token
                        ctc_out = utils.ctc_collapse(seq, self.model.blank_token_id)
                    else:
                        # predict by CTC
                        ctc_out = decode(outputs[i][:input_lens[i], :], 1, self.model.blank_token_id)[0][0]

                    # ground truth
                    gr_truth = list(labels[i][:label_lens[i]])

                    # calculated edit distance and required operations
                    dist, opr = edit_distance(gr_truth, ctc_out)

                    # increment number of phones
                    total_seq += len(gr_truth)

                    # update number of operations
                    for op_type, ids in opr.items():
                        if op_type == 'substitutions':
                            for orig, replace in ids:
                                op_dict[orig]['substitutions'][replace] += 1
                                op_dict[idx]['total'] += 1
                        else:
                            for idx in ids:
                                op_dict[idx][op_type] += 1
                                op_dict[idx]['total'] += 1
                    edit_dist_batch += dist

                if status == 1:
                    break

                print("Done with:", k, '/', self.test_loader.num_egs)

        # Average out the losses and edit distance
        test_loss /= len(self.test_loader)
        edit_dist_batch /= total_seq

        print("Edit distance - %.4f %% , Loss: %.7f" % (edit_dist_batch * 100, test_loss))

        # Store in lists for keeping track of model performance
        self.edit_dist.append((edit_dist_batch, epoch))
        self.test_losses.append((test_loss, epoch))

        # if testing loss is minimum, store it as the 'best.pth' model, which is used for feature extraction
        # store only when doing train/test together i.e. mode is train
        # dump probabilities
        prob_insert, prob_del, prob_substi = np.zeros(num_ph), np.zeros(num_ph), np.zeros((num_ph, num_ph))

        if test_loss == min([x[0] for x in self.test_losses]) and self.mode == 'train':
            print("Best new model found!")
            self.model.save_model(True, epoch, self.train_losses, self.test_losses, self.edit_dist,
                                  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)
            # Calculate the probabilities of insertion, deletion and substitution
            for ph, data in op_dict.items():
                prob_insert[ph] = data['insertions'] / data['total']
                prob_del[ph] = data['deletions'] / data['total']
                prob_substi[ph] = data['substitutions'] / data['total']

        if self.mode == 'train':
            # Dump probabilities
            prob_dump_path = self.config['dir']['pickle'] + self.arch_name + '_probs_'+ str(epoch) + '.pkl'
            with open(prob_dump_path, 'wb') as f:
                pickle.dump((prob_insert, prob_del, prob_substi), f)
                print("Dumped probabilities")

        return edit_dist_batch

    def test_one(self, file_paths):
        """
        Called during feature extraction
        :param file_paths: list of file paths to input .wav file to be tested
        :return: predicted phone probabilities after softmax layer
        """
        features, lens = [], []
        for file_path in file_paths:
            # read .wav file
            feat = utils.read_wav(file_path, winlen=self.config['window_size'], winstep=self.config['window_step'],
                                  fbank_filt=self.config['n_fbank'], mfcc_filt=self.config['n_mfcc'])
            tsteps, hidden_dim = feat.shape
            # calculate log mel filterbank energies for complete file and reshape so that it can be passed through model
            features.append((feat, file_path))
            lens.append(tsteps)

        final = []

        self.model.eval()

        with torch.no_grad():
            for i, (feat, path) in enumerate(features):
                input_model = torch.from_numpy(np.array(feat)).float()[None, :, :]
                cur_len = torch.from_numpy(np.array(lens[i:i+1])).long()

                if self.cuda:
                    input_model = input_model.cuda()
                    cur_len = cur_len.cuda()

                # Pass through model
                output = self.model(input_model, cur_len).cpu().numpy()
                # Apply softmax
                # softmax = torch.nn.functional.softmax(output, dim=2).numpy()
                final.append((output, path))

        id_to_phone = {v[0]: k for k, v in self.model.phone_to_id.items()}

        return final, self.model.phone_to_id, id_to_phone

    # Test for each wav file in the folder and also compares with ground truth if .PHN file exists
    def test_folder(self, test_folder):

        for wav_file in sorted(os.listdir(test_folder)):

            # Read input test file
            wav_path = os.path.join(test_folder, wav_file)
            dump_path = wav_path[:-4] + '_pred.txt'

            # Read only wav
            if wav_file == '.DS_Store' or wav_file.split('.')[-1] != 'wav':  # or os.path.exists(dump_path):
                continue

            feat = utils.read_wav(wav_path, winlen=self.config['window_size'], winstep=self.config['window_step'],
                                  fbank_filt=self.config['n_fbank'], mfcc_filt=self.config['n_mfcc'])

            tsteps, hidden_dim = feat.shape
            # calculate log mel filterbank energies for complete file
            feat_log_full = np.reshape(feat, (1, tsteps, hidden_dim))
            lens = np.array([tsteps])
            # prepare tensors
            inputs, lens = torch.from_numpy(np.array(feat_log_full)).float(), torch.from_numpy(np.array(lens)).long()
            id_to_phone = {v[0]: k for k, v in self.model.phone_to_id.items()}

            self.model.eval()

            with torch.no_grad():

                if self.cuda:
                    inputs = inputs.cuda()
                    lens = lens.cuda()

                # Pass through model
                outputs = self.model(inputs, lens).cpu().numpy()
                # Since only one example per batch and ignore blank token
                outputs = outputs[0]
                softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]
                # Take argmax to generate final string
                argmaxed = np.argmax(outputs, axis=1)
                # collapse according to CTC ruls
                final_str = utils.ctc_collapse(argmaxed, self.model.blank_token_id)
                ans = [id_to_phone[a] for a in final_str]
                # Generate dumpable format of phone, start time and end time
                print("Predicted:", ans)

            phone_path = wav_path[:-3] + 'PHN'

            # If .PHN file exists, report edit distance
            if os.path.exists(phone_path):
                truth = read_phones(phone_path, self.replacement)
                edit_dist, ops = edit_distance(truth, ans)
                print("Ground Truth:", truth, '\nEdit dsitance:', edit_dist)

                with open(dump_path, 'w') as f:
                    f.write('Predicted:\n')
                    f.write(' '.join(ans))
                    f.write('\nGround Truth:\n')
                    f.write(' '.join(truth))
                    f.write('\nEdit distance: ' + str(edit_dist))

            else:
                with open(dump_path, 'w') as f:
                    f.write('Predicted:\n')
                    f.write(' '.join(ans))

    def plot_loss_acc(self, epoch):
        """
        take train/test loss and test accuracy input and plot it over time
        :param epoch: to track performance across epochs
        """

        plt.clf()
        plt.plot([x[1] for x in self.train_losses], [x[0] for x in self.train_losses], c='r', label='Train')
        plt.plot([x[1] for x in self.test_losses], [x[0] for x in self.test_losses], c='b', label='Test')
        plt.title("Train/Test loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        filename = self.plots_dir + 'loss' + '_' + str(epoch) + '.png'
        plt.savefig(filename)

        plt.clf()
        plt.plot([x[1] for x in self.edit_dist], [x[0] for x in self.edit_dist], c='r')
        plt.title("Edit distance")
        plt.xlabel("Epochs")
        plt.ylabel("Edit distance")
        plt.grid(True)

        filename = self.plots_dir + 'test_acc' + '_' + str(epoch) + '.png'
        plt.savefig(filename)

        print("Saved plots")


def edit_distance(s1, s2):
    """
    Score for converting s1 into s2. Both s1 and s2 is a vector of phone IDs and not phones
    :param s1: string 1
    :param s2: string 2
    :return: edit distance and insert, delete and substitution probabilities
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1))

    op_dict = {}

    for i in range(m + 1):
        op_dict[i] = {}
        for j in range(n + 1):
            op_dict[i][j] = {'matches': [], 'insertions': [], 'deletions': [], 'substitutions': []}

    for i in range(m + 1):
        for j in range(n + 1):

            if i == 0:
                dp[i][j] = j
                op_dict[i][j]['insertions'] = s2[:j]
            elif j == 0:
                dp[i][j] = i
                op_dict[i][j]['deletions'] = s1[:i]
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op_dict[i][j] = deepcopy(op_dict[i - 1][j - 1])
                op_dict[i][j]['matches'].append(s1[i - 1])
            else:
                best = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
                dp[i][j] = best

                if best == dp[i - 1][j] + 1:
                    op_dict[i][j] = deepcopy(op_dict[i - 1][j])
                    op_dict[i][j]['deletions'].append(s1[i - 1])
                elif best == dp[i][j - 1] + 1:
                    op_dict[i][j] = deepcopy(op_dict[i][j - 1])
                    op_dict[i][j]['insertions'].append(s2[j - 1])
                else:
                    op_dict[i][j] = deepcopy(op_dict[i - 1][j - 1])
                    op_dict[i][j]['substitutions'].append((s1[i - 1], s2[j - 1]))

    return dp[m][n], op_dict[m][n]


def read_phones(phone_file_path, replacement):
    """
    Read .PHN file and return a compressed sequence of phones
    :param phone_file_path: path of .PHN file
    :param replacement: phones which are to be collapsed
    :return: a list of (phone, start_frame, end_frame)
    """
    labels = []

    with open(phone_file_path, 'r') as f:
        a = f.readlines()

    for phone in a:
        s_e_i = phone[:-1].split(' ')  # start, end, phenome_name e.g. 0 5432 'aa'
        _, _, ph = int(s_e_i[0]), int(s_e_i[1]), s_e_i[2]
        # Collapse
        for father, son in replacement.items():
            if ph in son:
                ph = father
                break
        # Append to list
        labels.append(ph)

    return labels


if __name__ == '__main__':
    a = dl_model('train')
    a.train()
    # a = dl_model('test')
    # a.test()
    # a = dl_model('test_one')
    # a.test_one(['trial/SX36.wav', 'trial/SX233.wav'])
    # a.test_folder('trial/')
    # a = [1, 1, 2, 3, 4, 4, 5]
    # b = [1, 2, 2, 3, 4, 5]
    # print(edit_distance(a, b))
