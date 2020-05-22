import numpy as np
import torch

import utils


# Returns data during training/testing from the dumped pickle file by metadata.py
class generic_dataloader:
    """
    Reads pickle dump from metadata.py, filters out sentences which are too long and stores them in a list
    so that it can be passed to the LSTM model
    """

    def __init__(self, config_file, batch_size=None):

        self.config = config_file
        if batch_size is None:
            self.batch_size = config_file['test']['batch_size']
        else:
            self.batch_size = batch_size
        self.idx = 0

    def build_dataset(self, list_of_sent, phone_to_id, bound_lengths=True):

        # Separate lists which return feature vectors, labels and lens
        self.final_feat = []
        self.final_labels = []
        self.input_lens = []
        self.label_lens = []

        if bound_lengths:
            # Keep only those which are within a range
            lengths = np.array([len(x[0]) for x in list_of_sent])
            avg, std = np.mean(lengths), np.std(lengths)
            max_allowed = int(avg + std * self.config['std_multiplier'])
            list_of_sent = [x for x in list_of_sent if len(x[0]) <= max_allowed]
            print("Ignored", (len(lengths) - len(list_of_sent)) / len(lengths), "fraction of examples")

        # Sort them according to lengths
        list_of_sent.sort(key=lambda pair: len(pair[0]), reverse=True)

        feature_dim = self.config['n_mfcc'] + self.config['n_fbank']
        pad_id = phone_to_id['PAD']

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
                current_labels = [phone_to_id[cur_ph] for cur_ph in labels]
                current_labels += [pad_id] * padding_l

                self.final_feat.append(current_features)
                self.final_labels.append(np.array(current_labels))
                self.input_lens.append(len(features))
                self.label_lens.append(len(labels))

            list_of_sent = list_of_sent[self.batch_size:]

        self.num_egs = len(self.input_lens)
        print("Total examples are:", self.num_egs)

    def return_batch(self, cuda):

        inputs = torch.from_numpy(np.array(self.final_feat[self.idx:self.idx + self.batch_size])).float()
        labels = torch.from_numpy(np.array(self.final_labels[self.idx:self.idx + self.batch_size])).long()
        input_lens = torch.from_numpy(np.array(self.input_lens[self.idx:self.idx + self.batch_size])).long()
        label_lens = torch.from_numpy(np.array(self.label_lens[self.idx:self.idx + self.batch_size])).long()

        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            input_lens = input_lens.cuda()
            label_lens = label_lens.cuda()

        self.idx += self.batch_size

        # Epoch ends if self.idx >= self.num_egs and hence return 1 which is detected by dl_model
        if self.idx >= self.num_egs:
            self.idx = 0
            return inputs, labels, input_lens, label_lens, 1
        else:
            return inputs, labels, input_lens, label_lens, 0

    def __len__(self):

        return (self.num_egs + self.batch_size - 1) // self.batch_size


class timit_dataloader(generic_dataloader):

    def __init__(self, type_, config_file):
        super().__init__(config_file)

        from metadata import timit_metadata
        metadata = timit_metadata(type_.upper(), config_file)
        # Returns huge list of feature vectors of audio recordings and phone sequences as tuples
        list_of_sent = metadata.gen_pickle()
        phone_to_id = utils.load_phone_mapping(config_file)

        self.build_dataset(list_of_sent, phone_to_id)


if __name__ == '__main__':

    from read_yaml import read_yaml
    config = read_yaml('config.yaml')
    a = timit_dataloader('test', config)
    print(a.return_batch())
