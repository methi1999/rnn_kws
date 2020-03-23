"""
Converts raw TIMIT data into a pickle dump which can be used during training
"""

import numpy as np
import pickle
import os
import utils
import json
from utils import listdir

class timit_data():

    def __init__(self, type_, config_file):

        self.config = config_file
        self.mode = type_
        self.db_path = config_file['dir']['dataset']

        # fold phones in list to the phone which is the key e.g. 'ao' is 'collapsed' into 'aa'
        self.replacement = utils.replacement_dict()

        feature_dim = self.config['n_fbank'] + self.config['n_mfcc']
        self.pkl_name = self.db_path + self.mode + '_rnn_ctc_' + str(feature_dim) + '.pkl'

        self.win_len, self.win_step = config_file['window_size'], config_file['window_step']

    # Generate and store pickle dump
    def gen_pickle(self):

        # Return if already exists
        if os.path.exists(self.pkl_name):
            print("Found pickle dump for", self.mode)
            with open(self.pkl_name, 'rb') as f:
                return pickle.load(f)

        print("Generating pickle dump for", self.mode)

        list_features, list_phones = [], []
        base_pth = self.db_path + self.mode
        all_phones = set()
        # Phone distribution is used to calculate weights
        num_distribution = {}

        # Iterate over entire dataset
        for dialect in sorted(listdir(base_pth)):

            print("Dialect:", dialect)

            for speaker_id in sorted(listdir(os.path.join(base_pth, dialect))):

                data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
                wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

                for wav_file in wav_files:

                    if wav_file in ['SA1.wav', 'SA2.wav']:
                        continue

                    wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)

                    final_vec = utils.read_wav(wav_path, winlen=self.config['window_size'],
                                               winstep=self.config['window_step'],
                                               fbank_filt=self.config['n_fbank'], mfcc_filt=self.config['n_mfcc'])

                    phenome_path = wav_path[:-3] + 'PHN'  # file which contains the phenome location data
                    # phones in current wav file
                    cur_phones = []

                    with open(phenome_path, 'r') as f:
                        a = f.readlines()

                    for phenome in a:
                        s_e_i = phenome[:-1].split(' ')  # start, end, phenome_name e.g. 0 5432 'aa'
                        start, end, ph = int(s_e_i[0]), int(s_e_i[1]), s_e_i[2]

                        # collapse into father phone
                        for father, list_of_sons in self.replacement.items():
                            if ph in list_of_sons:
                                ph = father
                                break
                        # update distribution
                        all_phones.add(ph)
                        if ph not in num_distribution.keys():
                            num_distribution[ph] = 0
                        num_distribution[ph] += 1

                        cur_phones.append(ph)

                    # Append current recording to the main list
                    list_features.append(final_vec)
                    list_phones.append(cur_phones)

        # Each item in to_return is a list corresponding to a single recording
        # Each recording is in turn a list of tuples of (ph, feature_vector) for each frame

        if self.mode == 'TRAIN':

            # Normalise feature vectors
            np_arr = np.concatenate(list_features, axis=0)
            print(np_arr.shape)
            np_mean = np.mean(np_arr, axis=0)
            np_std = np.std(np_arr, axis=0)
            # np_mean = np.zeros(np_mean.shape)
            # np_std = np.ones(np_std.shape)
            print("Mean:", np_mean, "\nStd. Dev:", np_std)

            # Weights are inversely proportional to number of phones encountered
            num_distribution = {k: 1 / v for k, v in num_distribution.items()}
            total_ph = sum(num_distribution.values())
            num_distribution = {k: v / total_ph for k, v in num_distribution.items()}
            # Dump mapping from id to phone. Used to convert NN output back to the phone it predicted
            phones_to_id = {}
            for ph in sorted(all_phones):
                phones_to_id[ph] = (len(phones_to_id), num_distribution[ph])

            phones_to_id['PAD'] = (len(phones_to_id), 0)
            # Dump this mapping
            fname = self.config['dir']['dataset'] + 'phone_mapping.json'
            with open(fname, 'w') as f:
                json.dump(phones_to_id, f)

        to_return = list(zip(list_features, list_phones))

        # Dump database
        with open(self.pkl_name, 'wb') as f:
            pickle.dump(to_return, f)
            print("Dumped pickle")

        return to_return


if __name__ == '__main__':
    config_file = {'dir': {'dataset': '../datasets/TEST/'}, 'feat_dim': 38}
    a = timit_data('TEST', config_file)
    a.gen_pickle()
