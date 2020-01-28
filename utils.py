import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import fbank, mfcc


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
    fbank_feat, energy = fbank(sig, samplerate=rate, winlen=winlen, winstep=winstep,
                               nfilt=fbank_filt, winfunc=np.hamming)
    # ***************Remove this********************
    fbank_feat = np.log(fbank_feat)
    # mfcc features
    mfcc_feat = mfcc(sig, samplerate=rate, winlen=winlen, winstep=winstep, numcep=mfcc_filt // 3, winfunc=np.hamming)
    mfcc_delta = np.concatenate((mfcc_feat[0:1, :], mfcc_feat[1:, :] - mfcc_feat[:-1, :]))
    mfcc_delta_delta = np.concatenate((mfcc_delta[0:1, :], mfcc_delta[1:, :] - mfcc_delta[:-1, :]))

    features = np.concatenate((fbank_feat, mfcc_feat, mfcc_delta, mfcc_delta_delta), axis=1)
    # features = np.concatenate((mfcc_feat, mfcc_delta, mfcc_delta_delta), axis=1)

    return features


def compress_seq(data):
    """
    Compresses a sequence (a,a,b,b,b,b,c,d,d....) into [(a,0,1),(b,2,5),...] i.e. [(phone, start_id, end_index]
    :param data: list of elements
    :return: data in the above format
    """

    final = []
    current_ph, current_start_idx = data[0], 0

    for i in range(2, len(data)):
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


def ctc_collapse(data, blank_id):
    """
    Collapse consecutive frames and then remove blank tokens
    :param data: list of elements e.g. [1,1,2,3,2,0,2,0]
    :param blank_id: blank token id
    :return: [1,2,3,2,2] in the above case
    """
    final = []
    current_ph = data[0]

    for i in range(2, len(data)):
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
