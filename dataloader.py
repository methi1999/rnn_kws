import numpy as np
import random
import json
from metadata import timit_data

# Set the seed to replicate results
random.seed(a=7)


# Returns data during training/testing from the dumped pickle file by metadata.py
class timit_loader():
    """
    Reads pickle dump from metadata.py, filters out sentences which are too long and stores them in a list
    so that it can be passed to the LSTM model
    """

    def __init__(self, type_, config_file):

        self.config = config_file

        metadata = timit_data(type_.upper(), config_file)
        # Returns huge list of feature vectors of audio recordings and phones as a tuple for each frame
        list_of_sent = metadata.gen_pickle()

        self.mode = type_  # train/test/test-one
        self.batch_size = config_file[type_]['batch_size']
        self.idx = 0

        # Load mapping
        try:
            fname = config_file['dir']['dataset'] + 'lstm_mapping.json'
            with open(fname, 'r') as f:
                self.phone_to_id = json.load(f)
            print("Phones and weights:", self.phone_to_id)

            self.weights = np.array([x[1] for x in self.phone_to_id.values()])

            assert len(self.phone_to_id) == config_file['num_phones'] + 1  # 1 for pad token

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

        sent_lens = [len(x[0]) for x in list_of_sent]
        label_lens = [len(x[1]) for x in list_of_sent]
        max_l = max(sent_lens)
        max_label_len = max(label_lens)
        print("Max length:", max_l, max_label_len, "; Ignored", (len(lengths) - len(sent_lens)) / len(lengths),
              "fraction of examples")

        feature_dim = self.config['feat_dim']
        pad_id = len(self.phone_to_id) - 1

        for sentence in list_of_sent:
            # Append 0s to feature vector to make a fixed dimensional matrix
            current_features = np.array(sentence[0])
            padding_l = max_l - current_features.shape[0]
            current_features = np.append(current_features, np.zeros((padding_l, feature_dim)), axis=0)
            # Add pad token for 0s
            padding_l = max_label_len - len(sentence[1])
            current_labels = [self.phone_to_id[cur_ph][0] for cur_ph in sentence[1]]
            current_labels += [pad_id] * padding_l

            self.final_feat.append(current_features)
            self.final_labels.append(np.array(current_labels))
            self.input_lens.append(len(sentence[0]))
            self.label_lens.append(len(sentence[1]))

        # Sort them according to lengths
        zipped = list(zip(self.final_feat, self.final_labels, self.input_lens, self.label_lens))
        zipped.sort(key=lambda triplet: triplet[2], reverse=True)

        self.final_feat, self.final_labels = [x[0] for x in zipped], [x[1] for x in zipped]
        self.input_lens, self.label_lens = [x[2] for x in zipped], [x[3] for x in zipped]

        self.num_egs = len(self.input_lens)
        print("Total examples for", self.mode, "are:", self.num_egs)

    def return_batch(self):

        inputs = self.final_feat[self.idx:self.idx + self.batch_size]
        labels = self.final_labels[self.idx:self.idx + self.batch_size]
        input_lens = self.input_lens[self.idx:self.idx + self.batch_size]
        label_lens = self.label_lens[self.idx:self.idx + self.batch_size]

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
    config_file = {'dir': {'dataset': '../datasets/TIMIT/'}, 'feat_dim': 26, 'num_phones': 39, 'std_multiplier': 4,
                   'test': {'batch_size': 1}}
    a = timit_loader('test', config_file)
