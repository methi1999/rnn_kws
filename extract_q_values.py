import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import fbank
import torch
import os
import json
from read_yaml import read_yaml
from dl_model import dl_model
import pickle
from hypo_search import generate_lattice, traverse_best_lattice, find_q_values

# Set the seed to replicate results
np.random.seed(7)

# fold phones in list to the phone which is the key e.g. 'ao' is 'collapsed' into 'aa'
replacement = {'aa': ['ao'], 'ah': ['ax', 'ax-h'], 'er': ['axr'], 'hh': ['hv'], 'ih': ['ix'],
               'l': ['el'], 'm': ['em'], 'n': ['en', 'nx'], 'ng': ['eng'], 'sh': ['zh'],
               'pau': ['pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl', 'h#', 'epi', 'q'],
               'uw': ['ux']}


# Ignore DS_Store files found on Mac
def listdir(pth):
    return [x for x in os.listdir(pth) if x != '.DS_Store']


class QValGenModel:
    """
    Loads the trained LSTM model for phone prediction and runs the chosen audio files through the model
    """

    def __init__(self, config, min_phones, recordings_dump_path, model_out_path):
        """
        :param config: config files
        :param min_phones: minimum number of instances of each phone to calculate Q value
        :param recordings_dump_path: path to dump the feature vectors of the recordings to be considered
        :param model_out_path: path to final q value dump
        """

        self.config = config
        self.pkl_name = recordings_dump_path
        self.outputs_path = model_out_path
        self.min_phones = min_phones
        self.idx = 0
        self.win_len, self.win_step = config['window_size'], config['window_step']
        # Initialise model
        self.rnn = dl_model('test_one')

        # Load mapping of phoen to id
        try:
            file_name = config['dir']['dataset'] + 'lstm_mapping.json'
            with open(file_name, 'r') as f:
                self.phone_to_id = json.load(f)
            self.phone_to_id = {k: v[0] for k, v in self.phone_to_id.items()}  # drop weight distribution
            print("Phones:", self.phone_to_id)

            assert len(self.phone_to_id) == config['num_phones'] + 1  # 1 for pad token

        except:
            print("Can't find phone mapping")
            exit(0)


    def gen_pickle(self):
        """
        # Iterates over the TEST dataset and picks up recordings such that each phone is covered atleast x no of times
        :return: Huge list of feature vectors of audio recordings and phones as a tuple for each frame
                 Each item in returned list is a list corresponding to a single recording
                 Each recording is in turn a list of tuples of (ph, feature_vector) for each frame
        """

        # Return if already exists
        if os.path.exists(self.pkl_name):
            print("Found pickle dump for recordings to be tested")
            with open(self.pkl_name, 'rb') as f:
                return pickle.load(f)

        print("Generating Q value pickle dump for a minimum of", self.min_phones, 'utterances of each phone')

        # final list to be returned
        to_return = []

        base_pth = self.config['dir']['dataset'] + 'TEST/'

        # keeps track of number of phones. Terminate only when all phones are above a threshold
        ph_count_dict = {}
        for phone in self.phone_to_id.keys():
            if phone != 'PAD':
                ph_count_dict[phone] = 0

        # keywords chosen
        keywords_chosen = set()

        paths = []

        # Iterate over entire dataset and store paths of wav files
        for dialect in sorted(listdir(base_pth)):

            for speaker_id in sorted(listdir(os.path.join(base_pth, dialect))):

                data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
                wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

                for wav_file in wav_files:
                    wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)
                    wrd_path = wav_path[:-3] + 'WRD'
                    phone_path = wrd_path[:-3] + 'PHN'

                    paths.append((wav_path, wrd_path, phone_path))

        # Shuffle the recordings so that we pick up recordings from various dialects and speakers
        np.random.shuffle(paths)

        for wav_path, wrd_path, phone_path in paths:

            # break if found required number of phones
            if all(x > self.min_phones for x in ph_count_dict.values()):
                print("Found enough utterances to cover all phones")
                break

            cur_phones = []

            with open(wrd_path, 'r') as f:
                wrd_list = f.readlines()
                for line in wrd_list:
                    # extract word from sframe, eframe, word format
                    word_start, word_end, word = line.split(' ')
                    word = word[:-1]
                    keywords_chosen.add(word)

            with open(phone_path, 'r') as f:
                phones_read = f.readlines()

            for phone in phones_read:
                s_e_i = phone[:-1].split(' ')  # start, end, phenome_name e.g. 0 5432 'aa'
                start, end, ph = int(s_e_i[0]), int(s_e_i[1]), s_e_i[2]

                # collapse into father phone
                for father, list_of_sons in replacement.items():
                    if ph in list_of_sons:
                        ph = father
                        break
                cur_phones.append(ph)
                # increment count of phone
                ph_count_dict[ph] += 1

            (rate, sig) = wav.read(wav_path)
            # sig ranges from -32768 to +32768 AND NOT -1 to +1
            feat, energy = fbank(sig, samplerate=rate, winlen=self.win_len, winstep=self.win_step,
                                 nfilt=self.config['feat_dim'], winfunc=np.hamming)

            feat_log_full = np.log(feat)  # calculate log mel filterbank energies for complete file
            to_return.append((feat_log_full, cur_phones))

        print("Final phone count dict:", ph_count_dict)
        with open(self.pkl_name, 'wb') as f:
            pickle.dump(to_return, f)
            print("Dumped pickle for recordings to be tested")

        print("Final chosen words:", keywords_chosen)

        return to_return

    def build_dataset(self, list_of_sent):
        # each element in list-of_sent is a tuple (ilterbank features of full sentence, list of phones)

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
            current_labels = [self.phone_to_id[cur_ph] for cur_ph in sentence[1]]
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

        self.batch_size = min(self.config['test']['batch_size'], len(self.final_feat))

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

    def get_outputs(self):
        """
        Run model through chosen recordings and dump the output
        :return: output probabilities along with ground truth labels and corresponding lengths
        """

        if os.path.exists(self.outputs_path):
            with open(self.outputs_path, 'rb') as f:
                print("Loaded database file from pickle dump")
                return pickle.load(f)

        # build dataset of sentences to be tested
        sent = self.gen_pickle()
        self.build_dataset(sent)

        self.rnn.model.eval()
        cuda = (self.config['cuda'] and torch.cuda.is_available())
        # final outputs
        final_outs = []
        cur_batch = 0
        total_egs = len(self.final_feat)

        while True:

            cur_batch += 1
            print("Batch:", cur_batch, '/', (total_egs + self.batch_size + 1) // self.batch_size)

            # Get batch of feature vectors, labels and lengths along with status (when to end epoch)
            inputs, labels, input_lens, label_lens, status = self.return_batch()
            inputs, labels, input_lens, label_lens = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(
                np.array(labels)).long(), torch.from_numpy(np.array(input_lens)).long(), torch.from_numpy(
                np.array(label_lens)).long()

            if cuda:
                inputs = inputs.cuda()
                input_lens = input_lens.cuda()
                label_lens = label_lens.cuda()

            # forward pass
            outputs = self.rnn.model(inputs, input_lens).detach().cpu().numpy()

            # softmax
            for i in range(outputs.shape[0]):
                softmax = np.exp(outputs[i]) / np.sum(np.exp(outputs[i]), axis=1)[:, None]
                final_outs.append((softmax, input_lens[i], labels[i], label_lens[i]))

            if status == 1:
                break

        with open(self.outputs_path, 'wb') as f:
            pickle.dump((final_outs, self.phone_to_id), f)
            print("Dumped model output")

        return final_outs, self.phone_to_id


def find_batch_q(dump_path, min_phones):
    """
    Computes the q-vale for each phone averaged over a specified number of instances
    :param dump_path: path to dump file
    :param min_phones: minimum number of phones to be covered
    :return: a dictionary of q-value for each phone and probabilities for insertion, deletion, substitution
    """

    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            vals = pickle.load(f)
            print('Loaded Q values from dump:', vals[0])
            return vals

    # minimum node probability to qualify as a candidate
    h_spike = 0.2

    config = read_yaml()

    if not os.path.exists(config['dir']['pickle']):
        os.mkdir(config['dir']['pickle'])

    database_name = config['dir']['pickle'] + 'QValGenModel_in_' + str(min_phones) + '.pkl'
    model_out_name = config['dir']['pickle'] + 'QValGenModel_out_' + str(min_phones) + '.pkl'

    # Instantiates the model to calculate predictions
    a = QValGenModel(config, min_phones, database_name, model_out_name)
    db, phone_to_id = a.get_outputs()

    # load probabilities vectors
    with open(config['dir']['pickle'] + 'probs.pkl', 'rb') as f:
        insert_prob, delete_prob, replace_prob = pickle.load(f)
        div = config['substi_prob_thesh_mult']
        temp = np.where(replace_prob == 0, 1, replace_prob)
        minimum = np.min(np.min(temp))
        print("Minimum substitution prob:", minimum)
        replace_prob = np.where(replace_prob == 0, minimum/div, replace_prob)
        print("Probabilities:\nInsert:", insert_prob, '\nDelete:', delete_prob, '\nSubsti:', replace_prob)

    final_dict = {}

    # for each sentence in database, find best subsequence, align and caluclate q values
    for i, (output, length, gr_phone, label_lens) in enumerate(db):
        print("On output:", str(i) + "/" + str(len(db)))
        cur_out = output[:length]
        gr_phone_ids = np.array(gr_phone[:label_lens])

        # Generate lattice from current predictions
        final_lattice = generate_lattice(cur_out, h_spike, True, a.rnn.model.blank_token_id, print_final_lattice=False)
        # Find best subsequence in lattice
        res = traverse_best_lattice(final_lattice, gr_phone_ids, insert_prob, delete_prob, replace_prob)
        # Calculate q values by comparing template and best match
        q_vals = find_q_values(gr_phone_ids, res, [x[0][1] for x in final_lattice],
                               insert_prob, delete_prob, replace_prob)

        # Add them up
        for ph_id, list_of_qvals in q_vals.items():
            if ph_id not in final_dict.keys():
                final_dict[ph_id] = []
            final_dict[ph_id] += list_of_qvals
    # Average out the values
    final_dict = {k: (sum(v) / len(v), len(v)) for k, v in final_dict.items()}

    with open(dump_path, 'wb') as f:
        pickle.dump((final_dict, insert_prob, delete_prob, replace_prob), f)

    print("Q values:", final_dict)
    return final_dict, insert_prob, delete_prob, replace_prob


if __name__ == '__main__':
    find_batch_q('pickle/final_q_vals.pkl', min_phones=75)
