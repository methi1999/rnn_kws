import numpy as np
import torch
import json

from metadata import timit_metadata


# Returns data during training/testing from the dumped pickle file by metadata.py
class timit_loader:
    """
    Reads pickle dump from metadata.py, filters out sentences which are too long and stores them in a list
    so that it can be passed to the LSTM model
    """

    def __init__(self, type_, config_file):

        self.config = config_file

        metadata = timit_metadata(type_.upper(), config_file)
        # Returns huge list of feature vectors of audio recordings and phones as tuples
        list_of_sent = metadata.gen_pickle()

        self.mode = type_  # train/test/test-one
        self.batch_size = config_file[type_]['batch_size']
        self.idx = 0

        # Load mapping
        try:
            fname = config_file['dir']['dataset'] + 'phone_mapping.json'
            with open(fname, 'r') as f:
                self.phone_to_id = json.load(f)

            assert len(self.phone_to_id) == config_file['num_phones']

        except:
            print("Can't find phone mapping")
            exit(0)

        # Fold phones and make list of training examples
        self.build_dataset(list_of_sent)

    def build_dataset(self, list_of_sent):

        # Separate lists which return feature vectors, labels and lens
        self.final_feat = []
        self.final_labels = []
        self.input_lens = []
        self.label_lens = []

        # Keep only those which are within a range
        lengths = np.array([len(x[0]) for x in list_of_sent])
        avg, std = np.mean(lengths), np.std(lengths)
        max_allowed = int(avg + std * self.config['std_multiplier'])
        list_of_sent = [x for x in list_of_sent if len(x[0]) <= max_allowed]
        print("Ignored", (len(lengths) - len(list_of_sent)) / len(lengths), "fraction of examples")

        # Sort them according to lengths
        list_of_sent.sort(key=lambda pair: len(pair[0]), reverse=True)

        feature_dim = self.config['n_mfcc'] + self.config['n_fbank']
        pad_id = len(self.phone_to_id)

        while len(list_of_sent):
            cur_slice = list_of_sent[:self.batch_size]
            max_l = max([len(x[0]) for x in cur_slice])
            max_label_len = max([len(x[1]) for x in cur_slice])

            for (features, labels) in cur_slice:
                # Append 0s to feature vector to make a fixed dimensional matrix
                current_features = np.array(features)
                padding_l = max_l - current_features.shape[0]
                current_features = np.append(current_features, np.zeros((padding_l, feature_dim)), axis=0)
                # Add pad token 0 to labels
                padding_l = max_label_len - len(labels)
                current_labels = [self.phone_to_id[cur_ph] for cur_ph in labels]
                current_labels += [pad_id] * padding_l

                self.final_feat.append(current_features)
                self.final_labels.append(np.array(current_labels))
                self.input_lens.append(len(features))
                self.label_lens.append(len(labels))

            list_of_sent = list_of_sent[self.batch_size:]

        self.num_egs = len(self.input_lens)
        print("Total examples for", self.mode, "are:", self.num_egs)

    def return_batch(self):

        inputs = torch.from_numpy(np.array(self.final_feat[self.idx:self.idx + self.batch_size])).float()
        labels = torch.from_numpy(np.array(self.final_labels[self.idx:self.idx + self.batch_size])).long()
        input_lens = torch.from_numpy(np.array(self.input_lens[self.idx:self.idx + self.batch_size])).long()
        label_lens = torch.from_numpy(np.array(self.label_lens[self.idx:self.idx + self.batch_size])).long()

        self.idx += self.batch_size

        # Epoch ends if self.idx >= self.num_egs and hence return 1 which is detected by dl_model
        if self.idx >= self.num_egs:
            self.idx = 0
            return inputs, labels, input_lens, label_lens, 1
        else:
            return inputs, labels, input_lens, label_lens, 0

    def __len__(self):

        return (self.num_egs + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':

    from read_yaml import read_yaml
    config = read_yaml('config.yaml')
    a = timit_loader('test', config)
    print(a.return_batch())
