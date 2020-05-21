import numpy as np
import scipy.io.wavfile as wav
from copy import deepcopy
from python_speech_features import mfcc, logfbank
import os
import json

# Ignore DS_Store files found on Mac
def listdir(pth):
    return [x for x in os.listdir(pth) if x != '.DS_Store']


def load_phone_mapping(config):
    try:
        file_name = config['dir']['dataset'] + 'phone_mapping.json'
        with open(file_name, 'r') as f:
            return json.load(f)

    except:
        print("Can't find phone mapping")
        exit(0)

def make_folder_if_dne(path):
    if not os.path.exists(path):
        os.mkdir(path)

def softmax(x):
    """
    Computes softmax for output of model
    :param x: x has shape (batch x time x number of classes)
    :return: softmax(x)
    """

    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)


def replacement_dict():
    return {'aa': ['ao'], 'ah': ['ax', 'ax-h'], 'er': ['axr'], 'hh': ['hv'], 'ih': ['ix'],
            'l': ['el'], 'm': ['em'], 'n': ['en', 'nx'], 'ng': ['eng'], 'sh': ['zh'],
            'pau': ['pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl', 'h#', 'epi', 'q'],
            'uw': ['ux']}


def collapse_phones(seq):
    replacement = replacement_dict()
    final = []
    for phone in seq:
        for father, sons in replacement.items():
            if phone in sons:
                phone = father
                break
        final.append(phone)
    return final


def read_wav(file_path, winlen, winstep, fbank_filt, mfcc_filt):
    (rate, sig) = wav.read(file_path)
    assert rate == 16000
    # sig ranges from -32768 to +32768 AND NOT -1 to +1

    features = None
    if fbank_filt:
        # log filterbank features
        logfbank_feat = logfbank(sig, samplerate=rate, winlen=winlen, winstep=winstep, nfilt=fbank_filt)
        if features is None:
            features = logfbank_feat
        else:
            features = np.concatenate((features, logfbank_feat), axis=1)
    if mfcc_filt:
        # mfcc features
        mfcc_feat = mfcc(sig, samplerate=rate, winlen=winlen, winstep=winstep, numcep=mfcc_filt // 3,
                         winfunc=np.hamming)
        mfcc_delta = np.concatenate((mfcc_feat[0:1, :], mfcc_feat[1:, :] - mfcc_feat[:-1, :]))
        mfcc_delta_delta = np.concatenate((mfcc_delta[0:1, :], mfcc_delta[1:, :] - mfcc_delta[:-1, :]))
        if features is None:
            features = np.concatenate((mfcc_feat, mfcc_delta, mfcc_delta_delta), axis=1)
        else:
            features = np.concatenate((features, mfcc_feat, mfcc_delta, mfcc_delta_delta), axis=1)

    return features


def edit_distance(s1, s2):
    """
    Score for converting s1 into s2. Both s1 and s2 is a vector of phone IDs and not phones
    :param s1: string 1
    :param s2: string 2
    :return: edit distance and insert, delete and substitution probabilities
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1))

    op_dict = {}

    for i in range(m + 1):
        op_dict[i] = {}
        for j in range(n + 1):
            op_dict[i][j] = {'matches': [], 'insertions': [], 'deletions': [], 'substitutions': []}

    for i in range(m + 1):
        for j in range(n + 1):

            if i == 0:
                dp[i][j] = j
                op_dict[i][j]['insertions'] = s2[:j]
            elif j == 0:
                dp[i][j] = i
                op_dict[i][j]['deletions'] = s1[:i]
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op_dict[i][j] = deepcopy(op_dict[i - 1][j - 1])
                op_dict[i][j]['matches'].append(s1[i - 1])
            else:
                best = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
                dp[i][j] = best

                if best == dp[i - 1][j] + 1:
                    op_dict[i][j] = deepcopy(op_dict[i - 1][j])
                    op_dict[i][j]['deletions'].append(s1[i - 1])
                elif best == dp[i][j - 1] + 1:
                    op_dict[i][j] = deepcopy(op_dict[i][j - 1])
                    op_dict[i][j]['insertions'].append(s2[j - 1])
                else:
                    op_dict[i][j] = deepcopy(op_dict[i - 1][j - 1])
                    op_dict[i][j]['substitutions'].append((s1[i - 1], s2[j - 1]))

    return dp[m][n], op_dict[m][n]


def read_PHN_file(phone_file_path):
    """
    Read .PHN file and return a compressed sequence of phones
    :param phone_file_path: path of .PHN file
    :param replacement: phones which are to be collapsed
    :return: a list of phones
    """
    labels = []

    with open(phone_file_path, 'r') as f:
        a = f.readlines()

    for phone in a:
        s_e_i = phone[:-1].split(' ')  # start, end, phone e.g. 0 5432 'aa'
        _, _, ph = int(s_e_i[0]), int(s_e_i[1]), s_e_i[2]
        # Collapse
        for father, son in utils.replacement_dict().items():
            if ph in son:
                ph = father
                break
        # Append to list
        labels.append(ph)

    return labels


def compress_seq(data):
    """
    Compresses a sequence (a,a,b,b,b,b,c,d,d....) into [(a,0,1),(b,2,5),...] i.e. [(phone, start_id, end_index]
    :param data: list of elements
    :return: data in the above format
    """

    final = []
    current_ph, current_start_idx = data[0], 0

    for i in range(1, len(data)):
        now_ph = data[i]
        if now_ph == current_ph:
            # same so continue
            continue
        else:
            # different element so append current and move on to the next
            final.append((current_ph, current_start_idx, i - 1))
            current_start_idx = i
            current_ph = now_ph
    # final element yet to be appended
    final.append((current_ph, current_start_idx, len(data) - 1))
    return final


def collapse_frames(data, blank_id):
    """
    Collapse consecutive frames and then remove blank tokens
    :param data: list of elements e.g. [1,1,2,3,2,0,2,0]
    :param blank_id: blank token id
    :return: [1,2,3,2,2] in the above case
    """
    final = []
    current_ph = data[0]

    for i in range(1, len(data)):
        now_ph = data[i]
        if now_ph == current_ph:
            continue
        else:
            final.append(current_ph)
            current_ph = now_ph

    # Append final element
    final.append(current_ph)
    # weed out the blank tokens
    final = [x for x in final if x != blank_id]
    return final


def frame_to_sample(frame_no, rate=16000, hop=10, window=25):
    if frame_no == 0:
        return 0
    multiplier = rate // 1000
    return multiplier * window + (frame_no - 1) * hop * multiplier


def sample_to_frame(num, is_start, rate=16000, window=25, hop=10):
    multi = rate // 1000
    if num < window * multi:
        return 0
    else:
        base_frame = (num - multi * window) // (multi * hop) + 1
        base_sample = frame_to_sample(base_frame)
        if is_start:
            if num - base_sample <= multi * hop // 2:
                return base_frame
            else:
                return base_frame + 1
        else:
            if num - base_sample <= multi * hop // 2:
                return base_frame - 1
            else:
                return base_frame
