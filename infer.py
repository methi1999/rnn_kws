import json
from read_yaml import read_yaml
import torch
from python_speech_features import fbank
from extract_q_values import find_batch_q
import pickle
import os
import operator
import numpy as np
from dl_model import dl_model
from hypo_search import generate_lattice, traverse_best_lattice, find_q_values
import scipy.io.wavfile as wav

replacement = {'aa': ['ao'], 'ah': ['ax', 'ax-h'], 'er': ['axr'], 'hh': ['hv'], 'ih': ['ix'],
               'l': ['el'], 'm': ['em'], 'n': ['en', 'nx'], 'ng': ['eng'], 'sh': ['zh'],
               'pau': ['pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl', 'h#', 'epi', 'q'],
               'uw': ['ux']}


# Ignore DS_Store files found on Mac
def listdir(pth):
    return [x for x in os.listdir(pth) if x != '.DS_Store']


def print_word_count(base_pth):
    """
    prints the count of each word in the TIMIT TEST dataset. Can be used to choose keywords to be tested
    :param base_pth: path to root directory TIMIT/TEST
    """
    keywords = {}

    for dialect in sorted(listdir(base_pth)):

        for speaker_id in sorted(listdir(os.path.join(base_pth, dialect))):

            data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
            wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

            for wav_file in wav_files:
                wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)
                wrd_path = wav_path[:-3] + 'WRD'

                with open(wrd_path, 'r') as f:
                    wrd_list = f.readlines()
                    for line in wrd_list:
                        # extract word from sframe, eframe, word format
                        word_start, word_end, word = line.split(' ')
                        word = word[:-1]
                        if word not in keywords.keys():
                            keywords[word] = 0
                        keywords[word] += 1
    keywords = sorted(keywords.items(), key=operator.itemgetter(1))
    print(keywords)


def gen_cases(base_pth, pkl_name, template_folder, num_templates, num_compares, keywords):
    """
    Generates test cases on which model is to be tested
    :param base_pth: root directory of TIMIT/TEST from where examples are picked
    :param pkl_name: path to pickle dump which stores list of paths
    :param template_folder: folder where template .wav clips are stored
    :param num_templates: templates per keyword
    :param num_compares: number of clips containing the keyword on which we want to test
    :param keywords: list of keywords to be tested
    :return: {kw1: {'templates':[[phone_list 1], [phone_list 2],..], 'test_wav_paths':[parth1,path2,...]}, kw2:...}
    """
    if os.path.exists(pkl_name):
        with open(pkl_name, 'rb') as f:
            return pickle.load(f)

    if not os.path.exists(template_folder):
        os.mkdir(template_folder)

    final_paths = {}

    paths = []

    for kw in keywords:
        final_paths[kw] = {'templates': [], 'test_wav_paths': []}

    for dialect in sorted(listdir(base_pth)):

        for speaker_id in sorted(listdir(os.path.join(base_pth, dialect))):

            data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
            wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

            for wav_file in wav_files:
                wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)
                wrd_path = wav_path[:-3] + 'WRD'
                phone_path = wrd_path[:-3] + 'PHN'

                paths.append((wav_path, wrd_path, phone_path))

    # shuffle paths
    np.random.shuffle(paths)

    for wav_path, wrd_path, phone_path in paths:

        with open(wrd_path, 'r') as f:
            wrd_list = f.readlines()

        with open(phone_path, 'r') as f:
            phone_list = f.readlines()

        for line in wrd_list:
            # extract word from start frame, end frame, word format
            word_start, word_end, word = line.split(' ')
            word_start, word_end = int(word_start), int(word_end)
            word = word[:-1]

            if word in keywords:
                # use this file as template
                if len(final_paths[word]['templates']) < num_templates:
                    list_of_phones = []

                    for phone in phone_list:
                        s_e_i = phone[:-1].split(' ')  # start, end, phenome_name e.g. 0 5432 'aa'
                        start, end, ph = int(s_e_i[0]), int(s_e_i[1]), s_e_i[2]

                        if start >= word_start and end <= word_end:
                            # collapse into father phone
                            for father, list_of_sons in replacement.items():
                                if ph in list_of_sons:
                                    ph = father
                                    break
                            list_of_phones.append(ph)

                    # append to list of templates
                    final_paths[word]['templates'].append(list_of_phones)
                    # trim the .wav file and store only the keyword part
                    dump_name = template_folder + word + '_' + str(len(final_paths[word]['templates'])) + '.wav'
                    (rate, sig) = wav.read(wav_path)
                    sig = sig[word_start:word_end + 1]
                    wav.write(dump_name, rate, sig)

                    print('Found keyword:', word, "in file", wav_path)

                # use wav file to compare with keyword
                elif len(final_paths[word]['test_wav_paths']) < num_compares:
                    final_paths[word]['test_wav_paths'].append(wav_path)

                else:
                    break

    with open(pkl_name, 'wb') as f:
        pickle.dump(final_paths, f)

    return final_paths


class BatchTestModel:

    def __init__(self, config, cases):
        """
        Loads the trained LSTM model for phone prediction and runs the chosen audio files through the model
        :param config: config file
        :param cases: the dictionary returned by gen_cases function
        """

        self.config = config
        self.idx = 0
        self.cases = cases
        self.win_len, self.win_step = config['window_size'], config['window_step']
        self.pkl_name = config['dir']['pickle'] + 'BatchTestModel_in.pkl'
        self.model_out_path = config['dir']['pickle'] + 'BatchTestModel_out.pkl'
        # Initialise model
        self.rnn = dl_model('test_one')

        # Load mapping
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
        # Iterates over the chosen cases of audio clips
        :return: list with each element as (feature_vectors, phones in sequence, keyword to be tested for)
        Note that phones in sequence is NOT strictly required. Included for further examination ONLY.
        Not used for prediction anywhere.
        """

        # Return if already exists
        if os.path.exists(self.pkl_name):
            print("Found pickle dump for recordings to be tested")
            with open(self.pkl_name, 'rb') as f:
                return pickle.load(f)

        paths = []

        # build list which contains .wav, .PHN paths
        for word, data in self.cases.items():

            data = data['test_wav_paths']
            for wav_path in data:
                paths.append((wav_path, wav_path[:-3] + 'PHN', word))

        to_return = []

        # append the feature vectors, ground truth phones (not used for prediction), keyword to be tested on
        for wav_path, phone_path, word in paths:
            cur_phones = []

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

            (rate, sig) = wav.read(wav_path)
            # sig ranges from -32768 to +32768 AND NOT -1 to +1
            feat, energy = fbank(sig, samplerate=rate, winlen=self.win_len, winstep=self.win_step,
                                 nfilt=self.config['feat_dim'], winfunc=np.hamming)

            feat_log_full = np.log(feat)  # calculate log mel filterbank energies for complete file
            to_return.append((feat_log_full, cur_phones, word))

        with open(self.pkl_name, 'wb') as f:
            pickle.dump(to_return, f)
            print("Dumped pickle for recordings to be tested")

        return to_return

    def build_dataset(self, list_of_sent):
        """
        Takes list of sentences and creates dataloader which can return data during testing
        :param list_of_sent: list of (feature_vectors, phones in sequence, keyword to be tested for)
        """

        # Separate lists which return feature vectors, labels and lens
        self.final_feat = []
        self.final_labels = []
        self.input_lens = []
        self.label_lens = []
        self.final_words = []

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

        for sentence in list_of_sent:
            # Append 0s to feature vector to make a fixed dimensional matrix
            current_features = np.array(sentence[0])
            padding_l = max_l - current_features.shape[0]
            current_features = np.append(current_features, np.zeros((padding_l, feature_dim)), axis=0)
            # Add pad token for 0s
            padding_l = max_label_len - len(sentence[1])
            current_labels = [self.phone_to_id[cur_ph] for cur_ph in sentence[1]]
            current_labels += [self.rnn.model.pad_token_id] * padding_l

            self.final_feat.append(current_features)
            self.final_labels.append(np.array(current_labels))
            self.input_lens.append(len(sentence[0]))
            self.label_lens.append(len(sentence[1]))
            self.final_words.append(sentence[2])

            # Sort them according to lengths
            zipped = list(zip(self.final_feat, self.final_labels, self.input_lens, self.label_lens, self.final_words))
            zipped.sort(key=lambda triplet: triplet[2], reverse=True)

            self.final_feat, self.final_labels = [x[0] for x in zipped], [x[1] for x in zipped]
            self.input_lens, self.label_lens, self.final_words = [x[2] for x in zipped], [x[3] for x in zipped], [x[4]
                                                                                                                  for x
                                                                                                                  in
                                                                                                                  zipped]

            self.num_egs = len(self.input_lens)

            self.batch_size = min(self.config['test']['batch_size'], len(self.final_feat))

    def return_batch(self):

        inputs = self.final_feat[self.idx:self.idx + self.batch_size]
        labels = self.final_labels[self.idx:self.idx + self.batch_size]
        input_lens = self.input_lens[self.idx:self.idx + self.batch_size]
        label_lens = self.label_lens[self.idx:self.idx + self.batch_size]
        words = self.final_words[self.idx:self.idx + self.batch_size]

        self.idx += self.batch_size

        # Epoch ends if self.idx >= self.num_egs and hence return 1 which is detected by dl_model
        if self.idx >= self.num_egs:
            self.idx = 0
            return inputs, labels, input_lens, label_lens, words, 1
        else:
            return inputs, labels, input_lens, label_lens, words, 0

    def get_outputs(self):
        """
        Run model through chosen recordings and dump the output
        :return: output probabilities, ground truth labels, corresponding lengths, keyword to be tested
        """

        if os.path.exists(self.model_out_path):
            with open(self.model_out_path, 'rb') as f:
                print("Loaded database file from pickle dump")
                return pickle.load(f)

        # generate database of audio clips to be tested
        sent = self.gen_pickle()
        self.build_dataset(sent)

        self.rnn.model.eval()
        cuda = (self.config['cuda'] and torch.cuda.is_available())
        final_outs = []
        cur_batch = 0
        total_egs = len(self.final_feat)

        while True:

            cur_batch += 1
            print("Batch:", cur_batch, '/', (total_egs + self.batch_size + 1) // self.batch_size)

            # Get batch of feature vectors, labels and lengths along with status (when to end epoch)
            inputs, labels, input_lens, label_lens, words, status = self.return_batch()
            inputs, labels, input_lens, label_lens = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(
                np.array(labels)).long(), torch.from_numpy(np.array(input_lens)).long(), torch.from_numpy(
                np.array(label_lens)).long()

            if cuda:
                inputs = inputs.cuda()
                input_lens = input_lens.cuda()
                label_lens = label_lens.cuda()

            outputs = self.rnn.model(inputs, input_lens).detach().cpu().numpy()

            # softmax and append desired objects to final_outs
            for i in range(outputs.shape[0]):
                softmax = np.exp(outputs[i]) / np.sum(np.exp(outputs[i]), axis=1)[:, None]
                final_outs.append((softmax, input_lens[i], labels[i], label_lens[i], words[i]))

            if status == 1:
                break

        with open(self.model_out_path, 'wb') as f:
            pickle.dump((final_outs, self.phone_to_id), f)
            print("Dumped model output")

        return final_outs, self.phone_to_id


def batch_test(num_templates, num_compares, pr_dump_path, results_dump_path):
    """
    Master function which carries out actual testing
    :param num_templates: number of templates for each keyword
    :param num_compares: number of clips in which each keyword needs to be searched for
    :param pr_dump_path: dump precision recall values
    :param results_dump_path: dump comparison results so that c values can be tweaked easily
    """

    if os.path.exists(results_dump_path):
        with open(results_dump_path, 'rb') as f:
            final_results = pickle.load(f)
    else:
        keywords = ['oily', 'people', 'before', 'living', 'potatoes', 'children', 'overalls', 'morning', 'enough', 'system',
                    'water', 'greasy', 'suit', 'dark', 'very', 'without', 'money']

        config = read_yaml()
        h_spike = config['h_spike']

        # Q values and probabilities are loaded. Important to load probability values from HERE since
        # they influence thresholds and Q-values
        thresholds, insert_prob, delete_prob, replace_prob = find_batch_q(config['dir']['pickle'] + 'final_q_vals.pkl',
                                                                          min_phones=50)

        pkl_name = config['dir']['pickle'] + 'test_cases_' + str(num_templates) + '_' + str(num_compares) + '.pkl'
        # generate cases to be tested on
        cases = gen_cases('../datasets/TIMIT/TEST/', pkl_name, 'templates/', num_templates, num_compares, keywords)

        # dictionary for storing c values required to declare keyword
        final_results = {}
        for kw in cases.keys():
            final_results[kw] = {'right': [], 'wrong': []}

        # initialise model
        a = BatchTestModel(config, cases)
        db, phone_to_id = a.get_outputs()

        assert len(phone_to_id) == replace_prob.shape[0] + 1

        # iterate over every clip and compare it with every template one-by-one
        # note that gr_phone_entire_clip is NOT USED
        for i, (output, length, gr_phone_entire_clip, label_lens, word_in_clip) in enumerate(db):
            print("On output:", str(i) + "/" + str(len(db)))
            cur_out = output[:length]

            # generate lattice from current predictions
            final_lattice = generate_lattice(cur_out, h_spike, True, a.rnn.model.blank_token_id, print_final_lattice=False)
            # node probabilites of full lattice. best only
            node_prob = [x[0][1] for x in final_lattice]
            # compare with every template
            for template_word, templates in cases.items():

                templates = templates['templates']

                for gr_phones in templates:
                    # template phone sequence
                    gr_phone_ids = [phone_to_id[x] for x in gr_phones]

                    res = traverse_best_lattice(final_lattice, gr_phone_ids, insert_prob, delete_prob, replace_prob)
                    # print("predicted:", [id_to_phone[x] for x in res], "actual:", [id_to_phone[x] for x in gr_phone_ids])
                    # calculate q values
                    q_vals = find_q_values(gr_phone_ids, res, node_prob, insert_prob, delete_prob, replace_prob)
                    # sum up the predicted q values
                    predicted_log_val, gr_log_val = 0, 0
                    for pred_phone, vals in q_vals.items():
                        for val in vals:
                            predicted_log_val += np.log(val)
                        gr_log_val += (np.log(thresholds[pred_phone][0]) * len(vals))

                    if template_word == word_in_clip:
                        # gr_log_val should be < predicted_log_val + c
                        final_results[template_word]['right'].append((gr_log_val, predicted_log_val))
                    else:
                        # gr_log_val should be > predicted_log_val + c
                        final_results[template_word]['wrong'].append((gr_log_val, predicted_log_val))

        with open(results_dump_path, 'wb') as f:
            pickle.dump(final_results, f)
            print("Dumped final results of testing")

    # grid search over parameter C
    cvals = [0.1, 0.3, 0.5, 0.7, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.3, 2.5, 3, 3.5, 4, 4.5]
    fscores = {}
    for c in cvals:
        fscores[c] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    for res in final_results.values():
        for gr, pred in res['right']:
            for c in cvals:
                if pred + c >= gr:
                    fscores[c]['tp'] += 1
                else:
                    fscores[c]['fn'] += 1
        for gr, pred in res['wrong']:
            for c in cvals:
                if pred + c >= gr:
                    fscores[c]['fp'] += 1
                else:
                    fscores[c]['tn'] += 1

    # store metrics in dictionary
    for c, vals in fscores.items():
        prec = vals['tp'] / (vals['tp'] + vals['fp'])
        recall = vals['tp'] / (vals['tp'] + vals['fn'])
        fscores[c]['prec-recall'] = (prec, recall, 2 * prec * recall / (prec + recall))

    print(fscores)
    with open(pr_dump_path, 'w') as f:
        json.dump(fscores, f, indent=4)
    print("Dumped JSON")


if __name__ == "__main__":
    batch_test(3, 12, 'pr.json', 'pickle/final_res.pkl')
    # print_word_count('../datasets/TIMIT/TEST/')