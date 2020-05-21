"""
The main driver file responsible for training, testing and extracting features
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pickle

from read_yaml import read_yaml
from dataloader import timit_dataloader
import utils
from beam_search import decode


class dl_model:

    def __init__(self, mode):

        # Read config fielewhich contains parameters
        self.config = read_yaml()
        self.mode = mode

        if self.config['rnn'] == 'liGRU':
            from architectures.ligru import liGRU as Model
        elif self.config['rnn'] == 'GRU' or self.config['rnn'] == 'LSTM':
            from architectures.rnn import RNN as Model
        elif self.config['rnn'] == 'TCN':
            from architectures.tcnn import TCN as Model
        elif self.config['rnn'] == 'BTCN':
            from architectures.tcnn import bidirectional_TCN as Model
        elif 'custom' in self.config['rnn']:
            from architectures.rnn import customRNN as Model
        else:
            Model = None
            print("Model import failed")
            exit(0)

        if 'custom' in self.config['rnn']:
            self.using_custom = True
        else:
            self.using_custom = False

        # Architecture name decides prefix for storing models and plots
        feature_dim = self.config['n_fbank'] + self.config['n_mfcc']
        self.arch_name = '_'.join(
            [self.config['rnn'], str(self.config['num_layers']), str(self.config['hidden_dim']), str(feature_dim)])

        print("Architecture:", self.arch_name)

        # Make folders if DNE
        utils.make_folder_if_dne(self.config['dir']['models'])
        utils.make_folder_if_dne(os.path.join(self.config['dir']['models'], self.arch_name))
        utils.make_folder_if_dne(self.config['dir']['plots'])
        utils.make_folder_if_dne(os.path.join(self.config['dir']['plots'], self.arch_name))
        utils.make_folder_if_dne(self.config['dir']['pickle'])
        utils.make_folder_if_dne(os.path.join(self.config['dir']['pickle'], self.arch_name))

        # load/initialise metrics to be stored and load model
        if mode == 'train':
            self.plots_dir = self.config['dir']['plots']
            # store hyperparameters
            self.total_epochs = self.config['train']['epochs']
            self.test_every = self.config['train']['test_every_epoch']
            self.test_per = self.config['train']['test_per_epoch']
            self.print_per = self.config['train']['print_per_epoch']
            self.save_every = self.config['train']['save_every']
            self.plot_every = self.config['train']['plot_every']

            # dataloader which returns batches of data
            self.train_loader = timit_dataloader('train', self.config)
            self.test_loader = timit_dataloader('test', self.config)
            # declare model
            self.model = Model(self.config, mode)

            self.start_epoch = 1
            self.edit_dist = []
            self.train_losses, self.test_losses = [], []
        elif mode == 'test':
            self.test_loader = timit_dataloader('test', self.config)
            # declare model
            self.model = Model(self.config, mode)
        else:
            # infer
            self.model = Model(self.config, mode)

        self.cuda = (self.config['use_cuda'] and torch.cuda.is_available())
        if self.cuda:
            self.model.cuda()

        # resume training from some stored model
        if self.mode == 'train' and self.config['train']['resume']:
            self.start_epoch, self.train_losses, self.test_losses, self.edit_dist = self.model.load_model(mode, self.arch_name)
            self.start_epoch += 1

        # load best model for testing/feature extraction
        elif self.mode == 'test' or mode == 'infer':
            self.model.load_model(mode, self.arch_name)

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

            if self.using_custom:
                dropout_mask_reset = [True] * (self.model.num_layers * (1 + self.config['bidirectional']))
            else:
                dropout_mask_reset = None

            try:
                print("Epoch:", str(epoch))
                epoch_loss = 0.0
                # i used for monitoring batch and printing loss, etc.
                i = 0
                while True:
                    i += 1

                    # Get batch of feature vectors, labels and lengths along with status (when to end epoch)
                    inputs, labels, input_lens, label_lens, status = self.train_loader.return_batch(self.cuda)
                    # print(input_lens, label_lens)
                    # zero the parameter gradients
                    self.model.optimizer.zero_grad()

                    # forward
                    if self.using_custom:
                        outputs = self.model(inputs, input_lens, dropout_mask_reset)
                        dropout_mask_reset = [False] * (self.model.num_layers * (1 + self.config['bidirectional']))
                    else:
                        outputs = self.model(inputs, input_lens)

                    # calculate loss
                    loss = self.model.calculate_loss(outputs, labels, input_lens, label_lens)
                    # backward
                    loss.backward()
                    # clip gradient
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    self.model.optimizer.step()

                    # store loss
                    epoch_loss += loss.item()

                    # print loss
                    if i in print_range and epoch == 1:
                        print('After %i batches, Current Loss = %.7f' % (i, epoch_loss / i))
                    elif i in print_range and epoch > 1:
                        print('After %i batches, Current Loss = %.7f, Avg. Loss = %.7f' % (
                                i, epoch_loss / i, np.mean(np.array([x[0] for x in self.train_losses]))))

                    # test model periodically
                    if i in test_range:
                        self.test(epoch)

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
                                          self.arch_name)

                if epoch % 5 == 0 and epoch < self.test_every:
                    self.test(epoch)
                elif epoch % self.test_every == 0:
                    self.test(epoch)
                # plot loss and accuracy
                if epoch % self.plot_every == 0:
                    self.plot_loss_acc(epoch)

            except KeyboardInterrupt:
                print("Saving model before quitting")
                if epoch > 1:
                    self.model.save_model(False, epoch-1, self.train_losses, self.test_losses, self.edit_dist,
                                          self.arch_name)
                exit(0)

    # test model
    def test(self, epoch=None):

        self.model.eval()
        # edit distance of batch
        edit_dist_batch = 0
        # number of sequences
        total_phones = 0
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

        num_sequences = 0
        # to_dump_probs, to_dump_labels = [], []

        with torch.no_grad():

            if self.using_custom:
                dropout_mask_reset = [True] * (self.model.num_layers * (1 + self.config['bidirectional']))
            else:
                dropout_mask_reset = None

            while True:

                # retrieve batch from dataloader
                inputs, labels, input_lens, label_lens, status = self.test_loader.return_batch(self.cuda)

                # zero the parameter gradients
                self.model.optimizer.zero_grad()

                # forward
                if self.using_custom:
                    outputs = self.model(inputs, input_lens, dropout_mask_reset)
                    dropout_mask_reset = [False] * (self.model.num_layers * (1 + self.config['bidirectional']))
                else:
                    outputs = self.model(inputs, input_lens)

                # calculate loss
                loss = self.model.calculate_loss(outputs, labels, input_lens, label_lens)
                test_loss += loss.item()

                outputs = outputs.cpu().numpy()
                labels = labels.cpu().numpy()

                num_sequences += outputs.shape[0]

                # calculate edit distance between ground truth and predicted sequence
                for i in range(outputs.shape[0]):
                    # predict by argmax
                    if decode_type == 'max':
                        # argmax over the phone channel
                        argmaxed = np.argmax(outputs, axis=2)
                        seq = list(argmaxed[i][:input_lens[i]])
                        # collapse neighbouring and remove blank token
                        output_seq = utils.collapse_frames(seq, self.model.blank_token_id)
                    else:
                        # predict by CTC
                        outputs = utils.softmax(outputs)
                        output_seq = decode(outputs[i, :input_lens[i], :], 1, self.model.blank_token_id)[0][0]

                    # ground truth
                    gr_truth = list(labels[i][:label_lens[i]])

                    # to_dump_probs.append(outputs[i][:input_lens[i], :])
                    # to_dump_labels.append(labels[i][:label_lens[i]])

                    # calculated edit distance and required operations
                    dist, opr = utils.edit_distance(gr_truth, output_seq)

                    # increment number of phones
                    total_phones += len(gr_truth)

                    # update number of operations
                    for op_type, ids in opr.items():
                        if op_type == 'substitutions':
                            for orig, replace in ids:
                                op_dict[orig]['substitutions'][replace] += 1
                                op_dict[orig]['total'] += 1
                        else:
                            for idx in ids:
                                op_dict[idx][op_type] += 1
                                op_dict[idx]['total'] += 1

                    edit_dist_batch += dist

                if status == 1:
                    break

                print("Done with:", num_sequences, '/', self.test_loader.num_egs)

        # Average out the losses and edit distance
        test_loss /= len(self.test_loader)
        edit_dist_batch /= total_phones

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
                                  self.arch_name)
            # Calculate the probabilities of insertion, deletion and substitution
            for ph, data in op_dict.items():
                prob_insert[ph] = data['insertions'] / data['total'] if data['total'] else 0
                prob_del[ph] = data['deletions'] / data['total'] if data['total'] else 0
                prob_substi[ph] = data['substitutions'] / data['total'] if data['total'] else 0

            # Dump best probability
            prob_dump_path = os.path.join(self.config['dir']['pickle'], self.arch_name, 'probs.pkl')
            with open(prob_dump_path, 'wb') as f:
                pickle.dump((prob_insert, prob_del, prob_substi), f)
                print("Dumped best probabilities")

        if self.mode == 'train':
            # Dump probabilities
            prob_dump_path = os.path.join(self.config['dir']['pickle'], self.arch_name, str(epoch)+'_probs.pkl')
            with open(prob_dump_path, 'wb') as f:
                pickle.dump((prob_insert, prob_del, prob_substi), f)
                print("Dumped probabilities")

        # with open('test_res.pkl', 'wb') as f:
        #     pickle.dump((to_dump_probs, to_dump_labels), f)
        self.model.train()

        return edit_dist_batch

    def infer(self, file_paths):

        self.model.eval()

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
            # compute feature vector for complete file and reshape so that it can be passed through model
            features.append((feat, file_path))
            lens.append(tsteps)

        final = []
        self.model.eval()

        with torch.no_grad():
            for i, (feat, path) in enumerate(features):
                print(i, '/', len(features))
                # prepare inputs for passing through model
                input_model = torch.from_numpy(np.array(feat)).float()[None, :, :]
                cur_len = torch.from_numpy(np.array(lens[i:i+1])).long()

                if self.cuda:
                    input_model = input_model.cuda()
                    cur_len = cur_len.cuda()

                # Pass through model
                output = self.model(input_model, cur_len).cpu().numpy()[0]
                # Apply softmax
                output = utils.softmax(output)
                final.append((output, path))

        return final, self.model.phone_to_id

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
                # softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]
                # Take argmax to generate final string
                argmaxed = np.argmax(outputs, axis=1)
                # collapse according to CTC rules
                final_str = utils.collapse_frames(argmaxed, self.model.blank_token_id)
                ans = [id_to_phone[a] for a in final_str]
                # Generate dumpable format of phone, start time and end time
                print("Predicted:", ans)

            phone_path = wav_path[:-3] + 'PHN'

            # If .PHN file exists, report edit distance
            if os.path.exists(phone_path):
                truth = utils.read_PHN_file(phone_path)
                edit_dist, ops = utils.edit_distance(truth, ans)
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
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot([x[1] for x in self.train_losses], [x[0] for x in self.train_losses], color='r', label='Train Loss')
        ax1.plot([x[1] for x in self.test_losses], [x[0] for x in self.test_losses], color='b', label='Test Loss')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel('PER')  # we already handled the x-label with ax1
        ax2.plot([x[1] for x in self.edit_dist], [x[0] for x in self.edit_dist], color='g', label='PER')
        ax2.tick_params(axis='y')
        ax2.legend(loc='upper right')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid(True)
        plt.legend()
        plt.title(self.arch_name)

        filename = os.path.join(self.plots_dir, self.arch_name, 'plot_' + str(epoch) + '.png')
        plt.savefig(filename)

        print("Saved plots")


if __name__ == '__main__':

    a = dl_model('train')
    a.train()

    # a = dl_model('test')
    # a.test()

    """
    Example usage for testing model on SA1 and SA2 sentences
    
    # declare model
    a = dl_model('infer')
    
    # store wav path in a list
    wav_paths, label_paths = [], []
    base_pth = '../datasets/TIMIT/TEST/'
    for dialect in sorted(utils.listdir(base_pth)):
        for speaker_id in sorted(utils.listdir(os.path.join(base_pth, dialect))):
            data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
            wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files
            for wav_file in wav_files:
                if wav_file in ['SA1.wav', 'SA2.wav']:
                    wav_paths.append(os.path.join(base_pth, dialect, speaker_id, wav_file))
                    label_paths.append(os.path.join(base_pth, dialect, speaker_id, wav_file[:-3]+'PHN'))
    
    # pass this list to model for inference                
    outputs, p_to_id, id_to_p = a.infer(wav_paths)
    # dump results
    with open('SA_res.pkl', 'wb') as f:
        pickle.dump((outputs, p_to_id, id_to_p), f)
    """