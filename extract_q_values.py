import numpy as np
import torch
import os
import pickle

from read_yaml import read_yaml
from dl_model import dl_model
from hypo_search import generate_lattice, traverse_best_lattice, find_q_values
import utils
from dataloader import generic_dataloader

# Set the seed to replicate results
np.random.seed(7)

# fold phones in list to the phone which is the key e.g. 'ao' is 'collapsed' into 'aa'
replacement = utils.replacement_dict()


class qval_metadata:
    """
    Loads the trained LSTM model for phone prediction and runs the chosen audio files through the model
    """

    def __init__(self, config, min_phones, recordings_dump_path):
        """
        :param config: config files
        :param min_phones: minimum number of instances of each phone to calculate Q value
        :param recordings_dump_path: path to dump the feature vectors of the recordings to be considered
        :param model_out_path: path to final q value dump
        """

        self.config = config
        self.pkl_name = recordings_dump_path
        self.min_phones = min_phones
        self.idx = 0
        self.win_len, self.win_step = config['window_size'], config['window_step']
        # Initialise model
        self.rnn = dl_model('infer')

        # Load mapping of phone to id
        self.phone_to_id = utils.load_phone_mapping(config)

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

        base_pth = self.config['dir']['dataset'] + 'TRAIN/'

        # keeps track of number of phones. Terminate only when all phones are above a threshold
        ph_count_dict = {}
        for phone, ph_id in self.phone_to_id.items():
            if ph_id < self.config['num_phones']:
                ph_count_dict[phone] = 0

        # keywords chosen
        keywords_chosen = set()

        paths = []

        # Iterate over entire dataset and store paths of wav files
        for dialect in sorted(utils.listdir(base_pth)):

            for speaker_id in sorted(utils.listdir(os.path.join(base_pth, dialect))):

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
                s_e_i = phone[:-1].split(' ')  # start, end, phonee_name e.g. 0 5432 'aa'
                start, end, ph = int(s_e_i[0]), int(s_e_i[1]), s_e_i[2]

                # collapse into father phone
                for father, list_of_sons in replacement.items():
                    if ph in list_of_sons:
                        ph = father
                        break
                cur_phones.append(ph)
                # increment count of phone
                ph_count_dict[ph] += 1

            final_vec = utils.read_wav(wav_path, winlen=self.config['window_size'], winstep=self.config['window_step'],
                                       fbank_filt=self.config['n_fbank'], mfcc_filt=self.config['n_mfcc'])
            to_return.append((final_vec, cur_phones))

        print("Final phone count dict:", ph_count_dict)
        with open(self.pkl_name, 'wb') as f:
            pickle.dump(to_return, f)
            print("Dumped pickle for recordings to be tested")

        print("Final chosen words:", keywords_chosen)

        return to_return


class qval_dataloader(generic_dataloader):

    def __init__(self, config_file, batch_size, min_phones, recordings_dump_path):
        super().__init__(config_file, batch_size)

        metadata = qval_metadata(config_file, min_phones, recordings_dump_path)
        ptoid = utils.load_phone_mapping(config_file)
        self.build_dataset(metadata.gen_pickle(), ptoid, bound_lengths=False)


class qval_model:

    def __init__(self, config, qval_dump_path, dataloader):

        self.config = config
        self.outputs_path = qval_dump_path
        self.rnn = dl_model('infer')
        self.dataloader = dataloader

        self.cuda = self.config['use_cuda'] and torch.cuda.is_available()

    def get_outputs(self):
        """
        Run model through chosen recordings and dump the output
        :return: output probabilities along with ground truth labels and corresponding lengths
        """

        if os.path.exists(self.outputs_path):
            with open(self.outputs_path, 'rb') as f:
                print("Loaded database file from pickle dump")
                return pickle.load(f)

        final_outs = []

        while True:

            # Get batch of feature vectors, labels and lengths along with status (when to end epoch)
            inputs, labels, input_lens, label_lens, status = self.dataloader.return_batch(self.cuda)

            # forward pass
            outputs = self.rnn.model(inputs, input_lens).detach().cpu().numpy()

            input_lens = input_lens.detach().cpu()
            labels = labels.detach().cpu()
            label_lens = label_lens.detach().cpu()

            # softmax
            softmax = utils.softmax(outputs)

            for i in range(softmax.shape[0]):
                final_outs.append((softmax[i], input_lens[i], labels[i], label_lens[i]))

            if status == 1:
                break

        with open(self.outputs_path, 'wb') as f:
            pickle.dump(final_outs, f)
            print("Dumped model output")

        return final_outs


def find_batch_q(dump_path, prob_path, min_phones, dec_type, top_n, exp_factor, min_sub_len=4, max_sub_len=15):
    """
    Computes the q-vale for each phone averaged over a specified number of instances
    :param max_sub_len: max length of random subsequence chosen from gr_phone for q-value calculation
    :param min_sub_len: min length of random subsequence chosen from gr_phone for q-value calculation
    :param prob_path: path to probability file
    :param dump_path: path to dump file
    :param min_phones: minimum number of phones to be covered
    :param dec_type: max or CTC
    :param top_n: top_n sequences to be considered
    :param exp_factor: weight assigned to probability score
    :return: a dictionary of q-value for each phone and probabilities for insertion, deletion, substitution
    """

    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            vals = pickle.load(f)
            print('Loaded Q values from dump:', vals[0])
            return vals

    config = read_yaml()
    phone_to_id = utils.load_phone_mapping(config)
    blank_token_id = phone_to_id['BLANK']

    if not os.path.exists(config['dir']['pickle']):
        os.mkdir(config['dir']['pickle'])

    database_name = config['dir']['pickle'] + 'QValGenModel_in_' + str(min_phones) + '.pkl'
    model_out_name = config['dir']['pickle'] + 'QValGenModel_out_' + str(min_phones) + '.pkl'

    # Instantiates the model to calculate predictions
    dataloader = qval_dataloader(config, config['test']['batch_size'], min_phones, database_name)
    model = qval_model(config, model_out_name, dataloader)

    db = model.get_outputs()

    # load probabilities vectors
    with open(prob_path, 'rb') as f:
        insert_prob, delete_prob, replace_prob = pickle.load(f)
        div = config['prob_thesh_const']

        temp = np.where(replace_prob == 0, 1, replace_prob)
        minimum = np.min(np.min(temp))
        print("Minimum substitution prob:", minimum)
        replace_prob = np.where(replace_prob == 0, minimum / div, replace_prob)

        temp = np.where(insert_prob == 0, 1, insert_prob)
        minimum = np.min(temp)
        print("Minimum insertion prob:", minimum)
        insert_prob = np.where(insert_prob == 0, minimum / div, insert_prob)

        temp = np.where(delete_prob == 0, 1, delete_prob)
        minimum = np.min(temp)
        print("Minimum deletion prob:", minimum)
        delete_prob = np.where(delete_prob == 0, minimum / div, delete_prob)

    final_dict = {}
    insert_prob_pow, delete_prob_pow, replace_prob_pow = np.power(insert_prob, exp_factor), \
                                                         np.power(delete_prob, exp_factor), \
                                                         np.power(replace_prob, exp_factor)

    print("Probabilities:\nInsert:", insert_prob, '\nDelete:', delete_prob, '\nSubsti:', replace_prob)

    # for each sentence in database, find best subsequence, align and calculate q values
    for i, (output, length, gr_phone, label_lens) in enumerate(db):
        print("On output:", str(i) + "/" + str(len(db)))
        cur_out = output[:length]
        gr_phone_ids = np.array(gr_phone[:label_lens])
        random_subsequence_len = np.random.randint(min_sub_len, max_sub_len)
        sub_start = np.random.randint(0, len(gr_phone_ids) - random_subsequence_len)
        random_subsequence = gr_phone_ids[sub_start:sub_start + random_subsequence_len]

        # Generate lattice from current predictions
        lattices = generate_lattice(cur_out, blank_token_id, dec_type, top_n)
        # Find best subsequence in lattice
        res_substring, best_lat = traverse_best_lattice(lattices, dec_type, random_subsequence,
                                                 insert_prob, delete_prob, replace_prob)
        # Calculate q values by comparing template and best match
        q_vals = find_q_values(random_subsequence, res_substring[0], res_substring[1],
                               insert_prob_pow, delete_prob_pow, replace_prob_pow)

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
    find_batch_q('pickle/final_q_vals.pkl', 'pickle/GRU_5_384_79/probs.pkl',
                 dec_type='max', min_phones=75, top_n=5, exp_factor=1)
