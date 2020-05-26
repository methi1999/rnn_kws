import torch
import numpy as np
import scipy.io.wavfile as wav
import pickle
import os
import time
import json

from dl_model import dl_model
from hypo_search import generate_lattice, traverse_best_lattice, find_q_values
from extract_q_values import find_batch_q
from read_yaml import read_yaml
import utils

replacement = utils.replacement_dict()
rnn_model = dl_model('infer')
np.random.seed(7)


def word_distribution(base_pth):
    words = {}

    for dialect in sorted(utils.listdir(base_pth)):

        for speaker_id in sorted(utils.listdir(os.path.join(base_pth, dialect))):

            data = sorted(utils.listdir(os.path.join(base_pth, dialect, speaker_id)))
            wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

            for wav_file in wav_files:
                wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)
                wrd_path = wav_path[:-3] + 'WRD'

                with open(wrd_path, 'r') as fw:
                    wrd_list = list(fw.readlines())

                for line in wrd_list:
                    # extract word from start sample, end sample, word format
                    word_start, word_end, word = line.rstrip().split(' ')
                    # add entry in dictionary
                    if word not in words.keys():
                        words[word] = 0
                    words[word] += 1
    print(sorted(words.items(), key=lambda x: x[1], reverse=True))


def choose_keywords(kw_base_pth, chosen_keywords, num_templates, gen_template, blank_id=None, template_save_loc=None):
    """
    Choose keywords from TIMIT TEST according to the minimum number of templates required
    :param blank_id: blank_id index
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

        for dialect in sorted(utils.listdir(kw_base_pth)):

            for speaker_id in sorted(utils.listdir(os.path.join(kw_base_pth, dialect))):

                data = sorted(utils.listdir(os.path.join(kw_base_pth, dialect, speaker_id)))
                wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

                for wav_file in wav_files:
                    wav_path = os.path.join(kw_base_pth, dialect, speaker_id, wav_file)
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
        outputs, phone_to_id, id_to_phone = rnn_model.infer(clip_paths)

        for out, path in outputs:
            word = path.split('/')[-1].split('_')[0]
            if word not in templates.keys():
                templates[word] = []

            out = np.argmax(out[0], axis=1)
            final_seq = utils.collapse_frames(out, blank_id)
            final_seq = [id_to_phone[x] for x in final_seq]
            if final_seq[0] == 'pau':
                final_seq = final_seq[1:]
            templates[word].append(final_seq)

        print("Templates from RNN:", templates)
        return templates

    else:
        print("Extracting templates from TIMIT")
        keywords = {}

        if isinstance(kw_base_pth, str):
            kw_base_pth = [kw_base_pth]

        for base_pth in kw_base_pth:

            for dialect in sorted(utils.listdir(base_pth)):

                for speaker_id in sorted(utils.listdir(os.path.join(base_pth, dialect))):

                    data = sorted(utils.listdir(os.path.join(base_pth, dialect, speaker_id)))
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


def gen_cases(base_pth_template, base_pth_totest, word_paths_pkl_name, num_templates, num_clips,
              num_none, keywords, gen_template):
    """
    Generates test cases on which model is to be tested
    :param base_pth_totest: path from where recordings are picked up
    :param gen_template: Whether to generate template using RNN or extract them from TIMIT
    :param base_pth_template: root directory of TIMIT/TEST from where examples are picked
    :param word_paths_pkl_name: path to pickle dump which stores list of paths
    :param num_clips: number of clips containing the keyword on which we want to test
    :param keywords: list of keywords to be tested
    :param num_templates: top-n templates to be returned for each keyword
    :param num_none: number of clips which do not contain any keyword
    :return: {kw1: {'templates':[[phone_list 1], [phone_list 2],..], 'test_wav_paths':[parth1,path2,...]}, kw2:...}
    """
    if os.path.exists(word_paths_pkl_name):
        with open(word_paths_pkl_name, 'rb') as f:
            return pickle.load(f)

    kws_chosen = choose_keywords(base_pth_template, keywords, num_templates, gen_template)
    final_paths = {}

    paths = []

    for kw in keywords:
        final_paths[kw] = {'templates': kws_chosen[kw], 'test_wav_paths': []}

    final_paths['NONE'] = {'templates': [], 'test_wav_paths': []}

    for dialect in sorted(utils.listdir(base_pth_totest)):

        for speaker_id in sorted(utils.listdir(os.path.join(base_pth_totest, dialect))):

            data = sorted(utils.listdir(os.path.join(base_pth_totest, dialect, speaker_id)))
            wav_files = [x for x in data if x.split('.')[-1] == 'wav']  # all the .wav files

            for wav_file in wav_files:
                wav_path = os.path.join(base_pth_totest, dialect, speaker_id, wav_file)
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

    with open(word_paths_pkl_name, 'wb') as f:
        pickle.dump(final_paths, f)

    print('Number of templates:', {word: len(dat['templates']) for word, dat in final_paths.items()})
    print('Number of clips:', {word: len(dat['test_wav_paths']) for word, dat in final_paths.items()})
    return final_paths


class test_metadata:

    def __init__(self, config, cases):
        """
        Loads the trained LSTM model for phone prediction and runs the chosen audio files through the model
        :param config: config file
        :param cases: the dictionary returned by gen_cases function
        """

        self.config = config
        self.cases = cases
        self.pkl_name = os.path.join(config['dir']['pickle'], rnn_model.arch_name, 'BatchTestModel_in.pkl')

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

            for wav_path in data['test_wav_paths']:
                paths.append((wav_path, wav_path[:-3] + 'PHN', word))

        to_return = []

        # append the feature vectors, ground truth phones (not used for prediction), keyword to be tested on
        for wav_path, phone_path, word in paths:
            cur_phones = []

            with open(phone_path, 'r') as f:
                phones_read = f.readlines()

            for phone in phones_read:
                s_e_i = phone[:-1].split(' ')  # start, end, phone_name e.g. 0 5432 'aa'
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


class test_dataloader:

    def __init__(self, config, cases):

        self.idx = 0
        self.config = config

        self.phone_to_id = utils.load_phone_mapping(config)

        metadata = test_metadata(config, cases)
        db = metadata.gen_pickle()
        self.build_dataset(db)

    def build_dataset(self, list_of_sent):
        """
        Takes list of sentences and creates dataloader which can return data during testing
        :param list_of_sent: list of (feature_vectors, phones in sequence, keyword to be tested for)
        """

        # Separate lists which return feature vectors, labels and lens
        self.final_feat = []
        self.final_labels = []
        self.input_lens = []
        self.final_words = []
        self.final_wavpath = []

        # Keep only those which are within a range
        lengths = np.array([len(x[0]) for x in list_of_sent])
        avg, std = np.mean(lengths), np.std(lengths)
        max_allowed = int(avg + std * self.config['std_multiplier'])
        list_of_sent = [x for x in list_of_sent if len(x[0]) <= max_allowed]

        sent_lens = [len(x[0]) for x in list_of_sent]
        max_l = max(sent_lens)

        feature_dim = self.config['n_mfcc'] + self.config['n_fbank']

        for (current_features, phones, word, wav_path) in list_of_sent:
            # Append 0s to feature vector to make a fixed dimensional matrix
            padding_l = max_l - current_features.shape[0]
            current_features = np.append(current_features, np.zeros((padding_l, feature_dim)), axis=0)

            self.final_feat.append(current_features)
            self.final_labels.append(phones)
            self.input_lens.append(len(current_features))
            self.final_words.append(word)
            self.final_wavpath.append(wav_path)

            # Sort them according to lengths
            zipped = list(zip(self.final_feat, self.final_labels, self.input_lens, self.final_words,
                              self.final_wavpath))
            zipped.sort(key=lambda triplet: triplet[2], reverse=True)

            self.final_feat, self.final_labels = [x[0] for x in zipped], [x[1] for x in zipped]
            self.input_lens = [x[2] for x in zipped]
            self.final_words, self.final_wavpath = [x[3] for x in zipped], [x[4] for x in zipped]

            self.num_egs = len(self.input_lens)

            self.batch_size = min(self.config['test']['batch_size'], len(self.final_feat))

    def return_batch(self):

        inputs = self.final_feat[self.idx:self.idx + self.batch_size]
        labels = self.final_labels[self.idx:self.idx + self.batch_size]
        input_lens = self.input_lens[self.idx:self.idx + self.batch_size]
        words = self.final_words[self.idx:self.idx + self.batch_size]
        paths = self.final_wavpath[self.idx:self.idx + self.batch_size]

        self.idx += self.batch_size

        # Epoch ends if self.idx >= self.num_egs and hence return 1 which is detected by dl_model
        if self.idx >= self.num_egs:
            self.idx = 0
            return inputs, labels, input_lens, words, paths, 1
        else:
            return inputs, labels, input_lens, words, paths, 0

    def __len__(self):

        return (self.num_egs + self.batch_size - 1) // self.batch_size


class test_model:

    def __init__(self, config, cases):

        self.model_out_path = os.path.join(config['dir']['pickle'], rnn_model.arch_name, 'BatchTestModel_out.pkl')
        self.config = config

        self.dataloader = test_dataloader(config, cases)

    def get_outputs(self):
        """
        Run model through chosen recordings and dump the output
        :return: output probabilities, ground truth labels, corresponding lengths, keyword to be tested
        """

        if os.path.exists(self.model_out_path):
            with open(self.model_out_path, 'rb') as f:
                print("Loaded database file from pickle dump")
                return pickle.load(f)

        cuda = self.config['use_cuda'] and torch.cuda.is_available()
        final_outs = []
        cur_batch = 0
        total_batches = len(self.dataloader)

        while True:

            cur_batch += 1
            print("Batch:", cur_batch, '/', total_batches)

            # Get batch of feature vectors, labels and lengths along with status (when to end epoch)
            inputs, labels, input_lens, words, wav_path, status = self.dataloader.return_batch()
            inputs, input_lens = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(
                np.array(input_lens)).long()

            if cuda:
                inputs = inputs.cuda()
                input_lens = input_lens.cuda()

            outputs = rnn_model.model(inputs, input_lens).detach().cpu().numpy()
            softmax = utils.softmax(outputs)

            # softmax and append desired objects to final_outs
            for i in range(softmax.shape[0]):
                final_outs.append((softmax[i], input_lens[i], labels[i], words[i], wav_path[i]))

            if status == 1:
                break

        with open(self.model_out_path, 'wb') as f:
            pickle.dump(final_outs, f)
            print("Dumped model output")

        return final_outs


def batch_test(config, dec_type, top_n, num_templates, num_compares, num_none, results_dump_path, exp_factor=1):
    """
    Master function which carries out actual testing
    :param dec_type: max or CTC
    :param top_n: top-n sequences are considered
    :param num_templates: number of templates for each keyword
    :param num_compares: number of clips in which each keyword needs to be searched for
    :param num_none: number of clips in which none of the keywords is present
    :param pr_dump_name: dump precision recall values
    :param results_dump_path: dump comparison results so that c values can be tweaked easily
    :param wrong_pred_path: path to folder where txt files are stored
    :param exp_factor: weight assigned to probability score
    """

    keywords = ['academic', 'reflect', 'equipment', 'program', 'rarely', 'national', 'social',
                'movies', 'greasy', 'water']
    # keywords = [
    #     'oily', 'people', 'before', 'living', 'potatoes', 'children', 'overalls', 'morning', 'enough', 'system',
    #     'water', 'greasy', 'suit', 'dark', 'very', 'without', 'money', 'reflect', 'program',
    #     'national', 'social', 'water', 'carry', 'time', 'before', 'always', 'often', 'people', 'money',
    #     'potatoes', 'children']
    # keywords = ['oily', 'people', 'before', 'living', 'water', 'children']
    # keywords = ['like', 'carry', 'will', 'potatoes', 'before', 'government', 'economic', 'overalls', 'through', 'money',
                # 'children']

    test_case_name = 'test_cases_' + str(num_templates) + '_' + str(num_compares) + '_' + str(num_none) + '.pkl'
    pkl_name = os.path.join(config['dir']['pickle'], rnn_model.arch_name, test_case_name)
    results_dump_path = os.path.join(config['dir']['pickle'], rnn_model.arch_name, results_dump_path)

    # generate cases to be tested on
    cases = gen_cases(['../datasets/TIMIT/TEST/', '../datasets/TIMIT/TRAIN/'], '../datasets/TIMIT/TEST/',
                      pkl_name, num_templates, num_compares, num_none, keywords, config['gen_template'])

    infer_mode = config['infer_mode']

    if os.path.exists(results_dump_path):
        with open(results_dump_path, 'rb') as f:
            return pickle.load(f)
    else:

        a = test_model(config, cases)

        # Q values and probabilities are loaded. Important to load probability values from HERE since
        # they influence thresholds and Q-values
        qval_pth = os.path.join(config['dir']['pickle'], rnn_model.arch_name, 'final_q_vals.pkl')
        prob_pth = os.path.join(config['dir']['pickle'], rnn_model.arch_name, 'probs.pkl')

        (thresholds, insert_prob, delete_prob, replace_prob) = find_batch_q(qval_pth, prob_pth, dec_type, top_n,
                                                                            exp_factor, rnn_model=rnn_model)

        # dictionary for storing c values required to declare keyword
        final_results = {}
        for kw in cases.keys():
            final_results[kw] = {}

        # initialise model
        db = a.get_outputs()
        phone_to_id = utils.load_phone_mapping(config)
        id_to_phone = {v: k for k, v in phone_to_id.items()}

        # iterate over every clip and compare it with every template one-by-one
        # note that gr_phone_entire_clip is NOT USED
        for i, (output, length, gr_phone_entire_clip, word_in_clip, wav_path) in enumerate(db):

            if i % (len(db) // 10) == 0:
                print("On output:", str(i) + "/" + str(len(db)))

            cur_out = output[:length]

            # generate lattice from current predictions
            lattices = generate_lattice(cur_out, rnn_model.model.blank_token_id, dec_type, top_n)
            # compare with every template
            for template_word, templates in cases.items():

                # if no keyword, then continue
                if template_word == 'NONE':
                    continue

                templates = templates['templates']
                final_results[template_word][i] = {'data': [], 'metadata': []}

                for template_phones in templates:
                    # template phone sequence
                    template_phone_ids = [phone_to_id[x] for x in template_phones]

                    (pred_phones, node_prob), final_lattice = traverse_best_lattice(lattices, dec_type,
                                                                                    template_phone_ids,
                                                                                    insert_prob, delete_prob,
                                                                                    replace_prob)
                    # out_for_cnn[word_in_clip].append((pred_phones, node_prob, word_in_clip == template_word))
                    # node probabilities of best lattice
                    substring_phones = [id_to_phone[x] for x in pred_phones]
                    final_lattice = [id_to_phone[x[0]] for x in final_lattice]

                    insert_prob_pow, delete_prob_pow, replace_prob_pow = np.power(insert_prob, exp_factor), \
                                                                         np.power(delete_prob, exp_factor), \
                                                                         np.power(replace_prob, exp_factor)

                    # calculate q values
                    q_vals = find_q_values(template_phone_ids, pred_phones, node_prob,
                                           insert_prob_pow, delete_prob_pow, replace_prob_pow)

                    metadata = (wav_path, word_in_clip, template_word, gr_phone_entire_clip, final_lattice,
                                substring_phones, template_phones)
                    final_results[template_word][i]['metadata'].append(metadata)

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
                            final_results[template_word][i]['data'].append(('right', above / total_phones))
                            # print('YES', above / total_phones)
                        else:
                            # gr_log_val should be > predicted_log_val + c
                            # print('NO', above / total_phones)
                            final_results[template_word][i]['data'].append(('wrong', above / total_phones))

                    else:
                        print("Infer Mode not defined")
                        exit(0)

        with open(results_dump_path, 'wb') as f:
            pickle.dump(final_results, f)
            print("Dumped final results of testing")

        return final_results

        # with open(cnn_dump_path, 'wb') as f:
        #     pickle.dump((out_for_cnn, cases), f)
        #     print("Dumped outputs for CNN training")


def calculate_p_r(config, final_results, pr_dump_name, wrong_pred_path, wrong_num):
    # grid search over parameter C

    pr_dump_path = os.path.join(config['dir']['pickle'], rnn_model.arch_name, pr_dump_name)

    infer_mode = config['infer_mode']
    if infer_mode == 'group':
        cvals = list(np.arange(0, 5, 0.1))
    elif infer_mode == 'indi':
        cvals = list(np.arange(0, 1, 0.05))
    else:
        print("Infer Mode not defined")
        exit(0)

    prec_recall_dat = {}
    for c in cvals:
        prec_recall_dat[c] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'prec_recall': ()}

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
                        wrong += metadata

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
                        wrong += metadata
                    else:
                        prec_recall_dat[c]['tn'] += 1

    # store metrics in dictionary
    fscore = []
    for c, vals in prec_recall_dat.items():
        prec = vals['tp'] / (vals['tp'] + vals['fp']) if (vals['tp'] + vals['fp']) else 0
        recall = vals['tp'] / (vals['tp'] + vals['fn']) if (vals['tp'] + vals['fn']) else 0
        if prec == 0 or recall == 0:
            prec_recall_dat[c]['prec_recall'] = (prec, recall, 0)
        else:
            prec_recall_dat[c]['prec_recall'] = (prec, recall, 2 * prec * recall / (prec + recall))
        fscore.append(2 * prec * recall / (prec + recall))

    # dump JSON
    print('Max F-score is', max(fscore))

    with open(pr_dump_path, 'w') as f:
        json.dump(prec_recall_dat, f, indent=4)
    print("Dumped JSON")

    # store incorrectly predicted metadata
    word_name_dict = {}

    if not os.path.exists(wrong_pred_path):
        os.mkdir(wrong_pred_path)

    np.random.shuffle(wrong)

    for data in wrong[:wrong_num]:

        (wav_path, word_in_clip, template_word, gr_phone_entire_clip, final_lattice, substring_phones, gr_phones) = data
        gr_phones = list(gr_phones)

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
        final_string += "Word present: " + word_in_clip + '\n'
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


def word_wise_p_r(config, final_results, testcases=None):

    infer_mode = config['infer_mode']
    keywords = list(final_results.keys())
    keywords.remove('NONE')

    if infer_mode == 'group':
        cvals = list(np.arange(0, 5, 0.1))
    elif infer_mode == 'indi':
        cvals = list(np.arange(0, 1, 0.05))
    else:
        print("Infer Mode not defined")
        exit(0)

    prec_recall_dat = {}

    for keyword in keywords:
        prec_recall_dat[keyword] = {}
        for c in cvals:
            prec_recall_dat[keyword][c] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'prec_recall': ()}

    # if any one of the templates match, declare keyword found, else not found
    for c in cvals:
        for word, res in final_results.items():
            for iteration, d in res.items():
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
                        prec_recall_dat[word][c]['tp'] += 1
                    else:
                        prec_recall_dat[word][c]['fn'] += 1

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
                        prec_recall_dat[word][c]['fp'] += 1
                    else:
                        prec_recall_dat[word][c]['tn'] += 1

    # store metrics in dictionary
    best_fscores = {}
    # print(prec_recall_dat)
    for kw, c_dict in prec_recall_dat.items():
        for c, vals in c_dict.items():
            prec = vals['tp'] / (vals['tp'] + vals['fp']) if (vals['tp'] + vals['fp']) else 0
            recall = vals['tp'] / (vals['tp'] + vals['fn']) if (vals['tp'] + vals['fn']) else 0
            if prec == 0 or recall == 0:
                prec_recall_dat[kw][c]['prec_recall'] = (prec, recall, 0)
            else:
                prec_recall_dat[kw][c]['prec_recall'] = (prec, recall, 2 * prec * recall / (prec + recall))

        best_c = 0
        best_fscore = 0
        for c in prec_recall_dat[kw]:
            if prec_recall_dat[kw][c]['prec_recall'][2] > best_fscore:
                best_fscore = prec_recall_dat[kw][c]['prec_recall'][2]
                best_c = c

        best_fscores[kw] = prec_recall_dat[kw][best_c]['prec_recall']

    print(best_fscores)

    if testcases is not None:
        lens = {}
        # print(testcases)
        for kw, data in testcases.items():
            if kw == 'NONE':
                continue
            temp_sum = sum([len(x) for x in data['templates']])
            lens[kw] = temp_sum/len(data['templates'])
        print("Average length of keywords:", lens)

        import matplotlib.pyplot as plt
        x, y = [], []
        for kw in keywords:
            x.append(lens[kw])
            y.append(best_fscores[kw][2])
        plt.xlabel("Average length of template")
        plt.ylabel("Best F-score")
        plt.grid(True)
        plt.scatter(x, y)
        plt.show()



def exp_grid_search(config):

    final = {}
    i = 0
    for exp in list(np.arange(0.2, 3.1, 0.1)):
        start = time.time()
        print("Starting at:", start)
        results = batch_test(config, 'max', 5, 3, 8, 170, 'final_res_' + str(i) + '.pkl', exp_factor=exp)
        final[exp] = calculate_p_r(config, results, 'pr_' + str(i) + '.json', 'incorrect/', wrong_num = 1000)
        print("Ended at:", time.time() - start)
        qvals_path = os.path.join('pickle', rnn_model.arch_name, 'final_q_vals.pkl')
        os.remove(qvals_path)
        i += 1

    print(max(final.values()), final)
    with open(os.path.join('pickle', rnn_model.arch_name, 'exp_grid_search.pkl'), 'wb') as f:
        pickle.dump(final, f)


if __name__ == "__main__":

    # word_distribution('../datasets/TIMIT/TEST/')

    config = read_yaml()
    # exp_grid_search(config)
    # res = pickle.load(open('pickle/GRU_5_384_79/final_res_exp_1.pkl', 'rb'))
    # cases = pickle.load(open('pickle/GRU_5_384_79/test_cases_3_8_170.pkl', 'rb'))
    # word_wise_p_r(config, res, cases)
    results = batch_test(config, 'max', 5, 3, 8, 170, 'final_res_exp_1_lattice1.pkl', exp_factor=1)
    fscore = calculate_p_r(config, results, 'pr_exp1_lattic1.json', 'incorrect/', wrong_num=1000)
    print(fscore)
    # batch_test('max', 3, 3, 20, 1000000, 'pickle/pr_full.json', 'pickle/pr_full.pkl', 'incorrect/')

    # a = dl_model('infer')
    # config = read_yaml()
    # path = 'trial/SX36.wav'
    # output, phone_to_id, id_to_phone = a.infer([path])
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
