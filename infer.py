import json
from read_yaml import read_yaml
import utils
import torch
from extract_q_values import find_batch_q
import scipy.io.wavfile as wav
import pickle
import os
import numpy as np
from dl_model import dl_model
from hypo_search import generate_lattice, traverse_best_lattice, find_q_values

replacement = utils.replacement_dict()


# Ignore DS_Store files found on Mac
def listdir(pth):
    return [x for x in os.listdir(pth) if x != '.DS_Store']


def choose_keywords(base_pth, chosen_keywords, num_templates, gen_template, template_save_loc=None, blank_id=40):
    """
    Choose keywords from TIMIT TEST according to the minimum number of templates required
    :param gen_template: If True, RNN is used for generating template also, else, template extracted form TIMIT
    :param template_save_loc: directory where keywords are stored
    :param base_pth: path to root directory TIMIT/TEST
    :param chosen_keywords: list of keywords to be tested on
    :param num_templates: the top-n templates which are chosen for every keyword
    """

    if gen_template:
        print("Generating templates using RNN")
        if not os.path.exists(template_save_loc):
            os.mkdir(template_save_loc)
        words = {}

        for dialect in sorted(listdir(base_pth)):

            for speaker_id in sorted(listdir(os.path.join(base_pth, dialect))):

                data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
                wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

                for wav_file in wav_files:
                    wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)
                    wrd_path = wav_path[:-3] + 'WRD'

                    with open(wrd_path, 'r') as fw:
                        wrd_list = list(fw.readlines())

                    for line in wrd_list:
                        # extract word from start sample, end sample, word format
                        word_start, word_end, word = line.rstrip().split(' ')
                        word_start, word_end = int(word_start), int(word_end)
                        # add entry in dictionary
                        if word not in words.keys() and word in chosen_keywords:
                            words[word] = []
                        if word in chosen_keywords:
                            words[word].append((wav_path, word_start, word_end))

        clip_paths = []
        for word, paths in words.items():
            np.random.shuffle(paths)
            i = 0
            for path, start, end in paths[:num_templates]:
                (rate, sig) = wav.read(path)
                assert rate == 16000
                sig = sig[start:end]
                write_name = template_save_loc + word + '_' + str(i) + '.wav'
                wav.write(write_name, rate, sig)
                clip_paths.append(write_name)
                i += 1

        templates = {}
        rnn = dl_model('test_one')
        outputs, phone_to_id, id_to_phone = rnn.test_one(clip_paths)

        for out, path in outputs:
            word = path.split('/')[-1].split('_')[0]
            if word not in templates.keys():
                templates[word] = []

            out = np.argmax(out[0], axis=1)
            final_seq = utils.ctc_collapse(out, blank_id)
            final_seq = [id_to_phone[x] for x in final_seq]
            if final_seq[0] == 'pau':
                final_seq = final_seq[1:]
            templates[word].append(final_seq)

        print("Templates from RNN:", templates)
        return templates

    else:
        print("Extracting templates from TIMIT")
        keywords = {}

        for dialect in sorted(listdir(base_pth)):

            for speaker_id in sorted(listdir(os.path.join(base_pth, dialect))):

                data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
                wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

                for wav_file in wav_files:
                    wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)
                    wrd_path = wav_path[:-3] + 'WRD'
                    ph_path = wav_path[:-3] + 'PHN'

                    with open(wrd_path, 'r') as fw:
                        wrd_list = list(fw.readlines())
                    with open(ph_path, 'r') as fp:
                        ph_list = list(fp.readlines())

                    for line in wrd_list:
                        phones_in_word = []
                        # extract word from start sample, end sample, word format
                        word_start, word_end, word = line.rstrip().split(' ')
                        word_start, word_end = int(word_start), int(word_end)
                        # add entry in dictionary
                        if word not in keywords.keys():
                            keywords[word] = {}
                        # iterate over list of phones
                        for ph_line in ph_list:
                            # extract phones from start sample, end sample, phone format
                            ph_start, ph_end, ph = ph_line.rstrip().split(' ')
                            ph_start, ph_end = int(ph_start), int(ph_end)
                            if ph_start == word_end:
                                break
                            # if phone corresponds to current word, add to list
                            if ph_start >= word_start and ph_end <= word_end:
                                # collapse
                                for father, list_of_sons in replacement.items():
                                    if ph in list_of_sons:
                                        ph = father
                                        break
                                phones_in_word.append(ph)

                        phones_in_word = tuple(phones_in_word)
                        # increment count in dictionary
                        if phones_in_word not in keywords[word].keys():
                            keywords[word][phones_in_word] = 0
                        keywords[word][phones_in_word] += 1

        # choose most frequently occurring templates from dictionary
        final_templates = {}
        for keyword in chosen_keywords:
            temps = keywords[keyword]
            temps = sorted(temps.items(), key=lambda kv: kv[1], reverse=True)
            chosen = [x[0] for x in temps[:num_templates]]
            final_templates[keyword] = chosen

        print("Templates from TIMIT:", final_templates)
        return final_templates


def gen_cases(base_pth, pkl_name, num_templates, num_clips, num_none, keywords, gen_template):
    """
    Generates test cases on which model is to be tested
    :param gen_template: Whether to generate template using RNN or extract them from TIMIT
    :param base_pth: root directory of TIMIT/TEST from where examples are picked
    :param pkl_name: path to pickle dump which stores list of paths
    :param num_clips: number of clips containing the keyword on which we want to test
    :param keywords: list of keywords to be tested
    :param num_templates: top-n templates to be returned for each keyword
    :param num_none: number of clips which do not contain any keyword
    :return: {kw1: {'templates':[[phone_list 1], [phone_list 2],..], 'test_wav_paths':[parth1,path2,...]}, kw2:...}
    """
    if os.path.exists(pkl_name):
        with open(pkl_name, 'rb') as f:
            return pickle.load(f)

    kws_chosen = choose_keywords(base_pth, keywords, num_templates, gen_template, template_save_loc='templates/')
    final_paths = {}

    paths = []

    for kw in keywords:
        final_paths[kw] = {'templates': kws_chosen[kw], 'test_wav_paths': []}

    final_paths['NONE'] = {'templates': [], 'test_wav_paths': []}

    for dialect in sorted(listdir(base_pth)):

        for speaker_id in sorted(listdir(os.path.join(base_pth, dialect))):

            data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
            wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

            for wav_file in wav_files:
                wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)
                wrd_path = wav_path[:-3] + 'WRD'

                paths.append((wav_path, wrd_path))

    # shuffle paths
    np.random.shuffle(paths)

    for wav_path, wrd_path in paths:

        with open(wrd_path, 'r') as f:
            wrd_list = f.readlines()

        for line in wrd_list:
            # extract word from start frame, end frame, word format
            word_start, word_end, word = line.rstrip().split(' ')

            if word in keywords:
                # use wav file to compare with keyword
                if len(final_paths[word]['test_wav_paths']) < num_clips:
                    final_paths[word]['test_wav_paths'].append(wav_path)
                break

            elif len(final_paths['NONE']['test_wav_paths']) < num_none:
                final_paths['NONE']['test_wav_paths'].append(wav_path)
                break

    with open(pkl_name, 'wb') as f:
        pickle.dump(final_paths, f)

    print('Number of templates:', {word: len(dat['templates']) for word, dat in final_paths.items()})
    print('Number of clips:', {word: len(dat['test_wav_paths']) for word, dat in final_paths.items()})
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
        total_length = 0

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

            final_vec = utils.read_wav(wav_path, winlen=self.config['window_size'], winstep=self.config['window_step'],
                                       fbank_filt=self.config['n_fbank'], mfcc_filt=self.config['n_mfcc'])

            to_return.append((final_vec, cur_phones, word, wav_path))

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
        self.final_wavpath = []

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

        feature_dim = self.config['n_mfcc'] + self.config['n_fbank']

        for sentence in list_of_sent:
            # Append 0s to feature vector to make a fixed dimensional matrix
            current_features = np.array(sentence[0])
            padding_l = max_l - current_features.shape[0]
            current_features = np.append(current_features, np.zeros((padding_l, feature_dim)), axis=0)
            # Add pad token for 0s
            padding_l = max_label_len - len(sentence[1])

            self.final_feat.append(current_features)
            self.final_labels.append(sentence[1])
            self.input_lens.append(len(sentence[0]))
            self.label_lens.append(len(sentence[1]))
            self.final_words.append(sentence[2])
            self.final_wavpath.append(sentence[3])

            # Sort them according to lengths
            zipped = list(zip(self.final_feat, self.final_labels, self.input_lens, self.label_lens, self.final_words,
                              self.final_wavpath))
            zipped.sort(key=lambda triplet: triplet[2], reverse=True)

            self.final_feat, self.final_labels = [x[0] for x in zipped], [x[1] for x in zipped]
            self.input_lens, self.label_lens = [x[2] for x in zipped], [x[3] for x in zipped]
            self.final_words, self.final_wavpath = [x[4] for x in zipped], [x[5] for x in zipped]

            self.num_egs = len(self.input_lens)

            self.batch_size = min(self.config['test']['batch_size'], len(self.final_feat))

    def return_batch(self):

        inputs = self.final_feat[self.idx:self.idx + self.batch_size]
        labels = self.final_labels[self.idx:self.idx + self.batch_size]
        input_lens = self.input_lens[self.idx:self.idx + self.batch_size]
        label_lens = self.label_lens[self.idx:self.idx + self.batch_size]
        words = self.final_words[self.idx:self.idx + self.batch_size]
        paths = self.final_wavpath[self.idx:self.idx + self.batch_size]

        self.idx += self.batch_size

        # Epoch ends if self.idx >= self.num_egs and hence return 1 which is detected by dl_model
        if self.idx >= self.num_egs:
            self.idx = 0
            return inputs, labels, input_lens, label_lens, words, paths, 1
        else:
            return inputs, labels, input_lens, label_lens, words, paths, 0

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
            inputs, labels, input_lens, label_lens, words, wav_path, status = self.return_batch()
            inputs, input_lens = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(
                np.array(input_lens)).long()

            if cuda:
                inputs = inputs.cuda()
                input_lens = input_lens.cuda()

            outputs = self.rnn.model(inputs, input_lens).detach().cpu().numpy()

            # softmax and append desired objects to final_outs
            for i in range(outputs.shape[0]):
                softmax = np.exp(outputs[i]) / np.sum(np.exp(outputs[i]), axis=1)[:, None]
                final_outs.append((softmax, input_lens[i], labels[i], label_lens[i], words[i], wav_path[i]))

            if status == 1:
                break

        with open(self.model_out_path, 'wb') as f:
            pickle.dump((final_outs, self.phone_to_id), f)
            print("Dumped model output")

        return final_outs, self.phone_to_id


def batch_test(dec_type, top_n, num_templates, num_compares, num_none, pr_dump_path, results_dump_path, wrong_pred_path,
               exp_factor=1):
    """
    Master function which carries out actual testing
    :param dec_type: max or CTC
    :param top_n: top-n sequences are considered
    :param num_templates: number of templates for each keyword
    :param num_compares: number of clips in which each keyword needs to be searched for
    :param num_none: number of clips in which none of the keywords is present
    :param pr_dump_path: dump precision recall values
    :param results_dump_path: dump comparison results so that c values can be tweaked easily
    :param wrong_pred_path: path to folder where txt files are stored
    :param exp_factor: weight assigned to probability score
    """
    config = read_yaml()
    keywords = ['academic', 'reflect', 'equipment', 'program', 'rarely', 'national', 'social', \
                'movies', 'greasy', 'water']
    pkl_name = config['dir']['pickle'] + 'test_cases_' + str(num_templates) + '_' + \
               str(num_compares) + '_' + str(num_none) + '.pkl'

    # generate cases to be tested on
    cases = gen_cases('../datasets/TIMIT/TRAIN/', pkl_name, num_templates, num_compares, num_none, keywords,
                      config['gen_template'])
    a = BatchTestModel(config, cases)
    a.gen_pickle()

    infer_mode = config['infer_mode']

    if os.path.exists(results_dump_path):
        with open(results_dump_path, 'rb') as f:
            final_results = pickle.load(f)
    else:
        # keywords = ['oily', 'people', 'before', 'living', 'potatoes', 'children', 'overalls', 'morning', 'enough',
        #             'system', 'water', 'greasy', 'suit', 'dark', 'very', 'without', 'money']
        keywords = ['academic', 'reflect', 'equipment', 'program', 'rarely', 'national', 'social', \
                    'movies', 'greasy', 'water']
        # keywords = ['oily', 'people', 'before', 'living', 'water', 'children']

        config = read_yaml()
        a = BatchTestModel(config, cases)

        # Q values and probabilities are loaded. Important to load probability values from HERE since
        # they influence thresholds and Q-values
        thresholds, insert_prob, delete_prob, replace_prob = find_batch_q(config['dir']['pickle'] + 'final_q_vals.pkl',
                                                                          config['dir'][
                                                                              'pickle'] + a.rnn.arch_name + '_probs.pkl',
                                                                          75,
                                                                          dec_type, top_n, exp_factor)

        # dictionary for storing c values required to declare keyword
        final_results = {}
        for kw in cases.keys():
            final_results[kw] = {}

        # initialise model
        db, phone_to_id = a.get_outputs()
        id_to_phone = {v: k for k, v in phone_to_id.items()}

        assert len(phone_to_id) == replace_prob.shape[0] + 1

        # iterate over every clip and compare it with every template one-by-one
        # note that gr_phone_entire_clip is NOT USED
        for i, (output, length, gr_phone_entire_clip, label_lens, word_in_clip, wav_path) in enumerate(db):
            print("On output:", str(i) + "/" + str(len(db)))
            cur_out = output[:length]

            # generate lattice from current predictions
            lattices = generate_lattice(cur_out, a.rnn.model.blank_token_id, dec_type, top_n)
            # compare with every template
            for template_word, templates in cases.items():

                # if no keyword, then continue
                if template_word == 'NONE':
                    continue

                templates = templates['templates']
                final_results[template_word][i] = {'data': [], 'metadata': ()}

                for gr_phones in templates:
                    # template phone sequence
                    gr_phone_ids = [phone_to_id[x] for x in gr_phones]

                    res_substring, final_lattice = traverse_best_lattice(lattices, dec_type, gr_phone_ids,
                                                                         insert_prob, delete_prob, replace_prob)
                    # node probabilities of best lattice
                    node_prob = [x[1] for x in final_lattice]
                    final_lattice = [id_to_phone[x[0]] for x in final_lattice]
                    substring_phones = [id_to_phone[x] for x in res_substring]
                    # calculate q values
                    q_vals = find_q_values(gr_phone_ids, res_substring, node_prob, insert_prob, delete_prob,
                                           replace_prob)
                    metadata = (
                        wav_path, template_word, gr_phone_entire_clip, final_lattice, substring_phones, gr_phones)

                    if infer_mode == 'group':
                        # sum up the predicted q values
                        predicted_log_val, gr_log_val = 0, 0

                        for pred_phone, vals in q_vals.items():
                            for val in vals:
                                predicted_log_val += np.log(val)
                            gr_log_val += (np.log(thresholds[pred_phone][0]) * len(vals))

                        if template_word == word_in_clip:
                            # gr_log_val should be < predicted_log_val + c
                            final_results[template_word][i]['data'].append(('right', gr_log_val, predicted_log_val))
                        else:
                            # gr_log_val should be > predicted_log_val + c
                            final_results[template_word][i]['data'].append(('wrong', gr_log_val, predicted_log_val))

                    elif infer_mode == 'indi':
                        above = 0
                        total_phones = 0
                        for pred_phone, vals in q_vals.items():
                            total_phones += len(vals)
                            for val in vals:
                                if val >= thresholds[pred_phone][0]:
                                    above += 1

                        if template_word == word_in_clip:
                            # gr_log_val should be < predicted_log_val + c
                            final_results[template_word][i]['data'].append(('right', above/total_phones))
                            # print('YES', above / total_phones)
                        else:
                            # gr_log_val should be > predicted_log_val + c
                            # print('NO', above / total_phones)
                            final_results[template_word][i]['data'].append(('wrong', above/total_phones))

                    else:
                        print("Infer Mode not defined")
                        exit(0)

                    # storing multiple results for each i corresponding to multiple templates
                    final_results[template_word][i]['metadata'] = metadata

        with open(results_dump_path, 'wb') as f:
            pickle.dump(final_results, f)
            print("Dumped final results of testing")

    # grid search over parameter C
    if config['infer_mode'] == 'group':
        cvals = list(np.arange(0, 5, 0.1))
        prec_recall_dat = {}
        for c in cvals:
            prec_recall_dat[c] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    elif config['infer_mode'] == 'indi':
        cvals = list(np.arange(0, 1, 0.05))
        prec_recall_dat = {}
        for c in cvals:
            prec_recall_dat[c] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    else:
        print("Infer Mode not defined")
        exit(0)

    # store incorect predictions
    wrong = []

    # if any one of the templates match, declare keyword found, else not found
    for c in cvals:
        for word, res in final_results.items():
            for iteration, d in res.items():
                metadata = d['metadata']
                egs = d['data']
                if egs[0][0] == 'right':
                    found = False

                    if infer_mode == 'group':
                        for _, gr, pred in egs:
                            if pred + c >= gr:
                                found = True
                    elif infer_mode == 'indi':
                        for _, ratio in egs:
                            if ratio >= c:
                                found = True
                    else:
                        print("Infer Mode not defined")
                        exit(0)

                    if found:
                        prec_recall_dat[c]['tp'] += 1
                    else:
                        prec_recall_dat[c]['fn'] += 1
                        if 2.5 >= c >= 1.5:
                            wrong.append(metadata)

                else:
                    found = False
                    if infer_mode == 'group':
                        for _, gr, pred in egs:
                            if pred + c >= gr:
                                found = True
                    elif infer_mode == 'indi':
                        for _, ratio in egs:
                            if ratio >= c:
                                found = True
                    else:
                        print("Infer Mode not defined")
                        exit(0)

                    if found:
                        prec_recall_dat[c]['fp'] += 1
                        if 2.5 >= c >= 1.5:
                            wrong.append(metadata)
                    else:
                        prec_recall_dat[c]['tn'] += 1

    # store metrics in dictionary
    fscore = []
    for c, vals in prec_recall_dat.items():
        prec = vals['tp'] / (vals['tp'] + vals['fp'])
        recall = vals['tp'] / (vals['tp'] + vals['fn'])
        prec_recall_dat[c]['prec-recall'] = (prec, recall, 2 * prec * recall / (prec + recall))
        fscore.append(2 * prec * recall / (prec + recall))

    # dump JSON
    print('Max F-score for exp_factor', exp_factor, 'is', max(fscore))
    with open(pr_dump_path, 'w') as f:
        json.dump(prec_recall_dat, f, indent=4)
    print("Dumped JSON")

    # store incorrectly predicted metadata
    word_name_dict = {}

    if not os.path.exists(wrong_pred_path):
        os.mkdir(wrong_pred_path)

    np.random.shuffle(wrong)

    for (wav_path, template_word, gr_phone_entire_clip, final_lattice, substring_phones, gr_phones) in wrong[:100]:

        if template_word not in word_name_dict:
            word_name_dict[template_word] = 0

        if len(gr_phone_entire_clip) > len(final_lattice):
            final_lattice += ['-'] * (len(gr_phone_entire_clip) - len(final_lattice))
        else:
            gr_phone_entire_clip += ['-'] * (len(final_lattice) - len(gr_phone_entire_clip))

        if len(substring_phones) > len(gr_phones):
            gr_phones += ['-'] * (len(substring_phones) - len(gr_phones))
        else:
            substring_phones += ['-'] * (len(gr_phones) - len(substring_phones))

        final_string = ".wav path: " + wav_path + '\n'
        final_string += "Looking for: " + template_word + '\n'
        final_string += "Ground truth lattice || Predicted Lattice\n"
        for gr, pred in zip(gr_phone_entire_clip, final_lattice):
            final_string += (gr + '\t\t\t\t' + pred + '\n')

        final_string += "\nTemplate || Best substring\n"
        for gr, pred in zip(gr_phones, substring_phones):
            final_string += (gr + '\t\t\t\t' + pred + '\n')

        fname = wrong_pred_path + template_word + '_' + str(word_name_dict[template_word]) + '.txt'
        with open(fname, 'w') as f:
            f.write(final_string)
        word_name_dict[template_word] += 1

    return max(fscore)


if __name__ == "__main__":
    final = {}
    i = 0
    for exp in list(np.arange(0.1, 3.1, 0.1)):
        final[exp] = batch_test('max', 5, 3, 8, 170, 'pickle/pr_' + str(i) + '.json',
                                'pickle/final_res_' + str(i) + '.pkl', 'incorrect/', exp)
    
        os.remove('pickle/final_q_vals.pkl')
        i += 1
    
    print(max(final.values()), final)
    with open('f.pkl', 'wb') as f:
        pickle.dump(final, f)

    # batch_test('max', 5, 3, 8, 170, 'pickle/pr_test.json', 'pickle/final_res_test.pkl', 'incorrect/')
    # batch_test('max', 3, 3, 2, 5, 'pickle/pr_test.json', 'pickle/pr_test.pkl', 'incorrect/')

    # a = dl_model('test_one')
    # config = read_yaml()
    # path = 'trial/SX36.wav'
    # output, phone_to_id, id_to_phone = a.test_one([path])
    # output = output[0][0][0]
    # output = np.exp(output) / np.sum(np.exp(output), axis=1)[:, None]
    # template = ['t', 'q', 'aa', 'r', 'dx', 'ih', 's', 'pau', 'ax']
    # template = utils.collapse_phones(template)
    # gr_phone_ids = [phone_to_id[x][0] for x in template]
    #
    # thresholds, insert_prob, delete_prob, replace_prob = find_batch_q(config['dir']['pickle'] + 'final_q_vals.pkl',
    #                                                                   config['dir'][
    #                                                                       'pickle'] + a.arch_name + '_probs.pkl',
    #                                                                   75, 'max', 3, 1)
    #
    # final_lattice = generate_lattice(output, a.model.blank_token_id, 'max', 3, print_final_lattice=True)
    # res, best_lattice = traverse_best_lattice(final_lattice, 'max', gr_phone_ids, insert_prob, delete_prob,
    #                                           replace_prob)
    # res_phones = [id_to_phone[x] for x in res]
    # print('Ground truth:', template, '\n', 'Predicted:', res_phones)
    # print(find_q_values(gr_phone_ids, res, [x[1] for x in best_lattice], insert_prob, delete_prob, replace_prob))
