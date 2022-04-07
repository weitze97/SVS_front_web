import numpy as np
from scipy.io.wavfile import read
import torch
import librosa
import json
import random
import pdb
import h5py
import os
import logging
import sys


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


# def load_wav_to_torch(full_path):
#     # sampling_rate, data = read(full_path)
#     # 自行指定載入的sample rate
#     data, sampling_rate = librosa.load(full_path, sr=22050)

#     # --------------------------------------------------------------
#     # new setting: melgan多的audio processing
#     # data = 0.95 * librosa.util.normalize(data)
#     # amplitude = np.random.uniform(low=0.3, high=1.0)
#     # data = data * amplitude
#     return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def phone2id(phone_seq):
    '''
    Converts a string of phonemes to a sequence of IDs
    '''

    all_phone = ["$0", "a", "ai", "an", "ang", "ao", 
    "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", 
    "f", "g", "h", "i", "ia", "ian", "iang", "iao", 
    "ie", "in", "ing", "iong", "irh", "iu", "j", "k", 
    "l", "m", "n", "o", "ong", "ou", "p", "q", "r", 
    "s", "sh", "t", "u", "ua", "uai", "uan", "uang", 
    "uei", "ui", "un", "uo", "v", "van", "ve", "vn", 
    "w", "wan", "wei", "wo", "wu", "x", "y", "yao", 
    "yi", "yin", "ying", "yong", "you", "yu", "z", "zh", "xx", "sil"]
    
    dict_phone2id = {}
    for idx, phone in enumerate(all_phone):
        dict_phone2id[phone] = idx + 1  # avoid zero, since zero is used for padding.
    
    id_seq = [dict_phone2id[phone] for phone in phone_seq]

    return id_seq, dict_phone2id


def type2id(phone_seq):
    init = ["$0",
    "b", "c", "ch", "d",
    "f", "g", "h", 
    "j", "k", 
    "l", "m", "n", "p", "q", "r", 
    "s", "sh", "t",
    "w",  "x", "y",
    "z", "zh"]

    final = ["a", "ai", "an", "ang", "ao", 
    "e", "ei", "en", "eng", "er", 
    "i", "ia", "ian", "iang", "iao", 
    "ie", "in", "ing", "iong", "irh", "iu",
    "o", "ong", "ou", 
    "u", "ua", "uai", "uan", "uang", 
    "uei", "ui", "un", "uo", "v", "van", "ve", "vn", 
    "wan", "wei", "wo", "wu", "yao", 
    "yi", "yin", "ying", "yong", "you", "yu"]

    zero_init = ["xx", "sil"]

    for i, pho in enumerate(phone_seq):
        if pho in init:
            phone_seq[i] = 1
        elif pho in final:
            phone_seq[i] = 2
        elif pho in zero_init:
            phone_seq[i] = 3
        else:
            print("Miss to define the type!")
            print(pho, len(pho))

    return phone_seq, 0
# def phone2id(word_seq):
#     '''
#     Converts a string of words to a sequence of IDs
#     '''
#     with open('filelists/all_phon', 'r') as f:
#         all_word = json.load(f)
    
#     dict_word2id = {}
#     for idx, word in enumerate(all_word):
#         dict_word2id[word] = idx + 1  # avoid zero, since zero is used for padding.
        
#     id_seq = [dict_word2id[word] for word in word_seq]

#     return id_seq, dict_word2id


def note2id(note_seq):
    '''
    Converts a string of notes to a sequence of IDs
    '''
    with open('taco2/filelists/f1/all_note_rb', 'r') as f:
        all_note = json.load(f)
    
    dict_note2id = {}
    for idx, note in enumerate(all_note):
        dict_note2id[note] = idx + 1  # avoid zero, since zero is used for padding.
    
    id_seq = [dict_note2id[note] for note in note_seq]

    return id_seq, dict_note2id

def notelen2id(note_seq):
    '''
    Converts a string of notes to a sequence of IDs
    '''
    with open('taco2/filelists/f1/all_notelen_rb', 'r') as f:
        all_note = json.load(f)
    
    dict_note2id = {}
    for idx, note in enumerate(all_note):
        dict_note2id[note] = idx + 1  # avoid zero, since zero is used for padding.
    
    id_seq = [dict_note2id[note] for note in note_seq]

    return id_seq, dict_note2id

def pl2fl(input_pl, pho_dur, sampling_rate, hop_length):
    frame_len = hop_length / sampling_rate
    pho_dur = [float(i) for i in pho_dur.split()]
    assert len(input_pl) == len(pho_dur)
    
    input_fl = []
    for phon, dur in zip(input_pl, pho_dur):
        num_pho_expanded = int(np.round(dur / frame_len))
        input_fl.extend([phon]*num_pho_expanded)
    input_fl = torch.IntTensor(input_fl)
    
    return input_fl


def onehot2phone(onehot_phone_seq):
    all_phone = ["$0", "a", "ai", "an", "ang", "ao", 
    "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", 
    "f", "g", "h", "i", "ia", "ian", "iang", "iao", 
    "ie", "in", "ing", "iong", "irh", "iu", "j", "k", 
    "l", "m", "n", "o", "ong", "ou", "p", "q", "r", 
    "s", "sh", "t", "u", "ua", "uai", "uan", "uang", 
    "uei", "ui", "un", "uo", "v", "van", "ve", "vn", 
    "w", "wan", "wei", "wo", "wu", "x", "y", "yao", 
    "yi", "yin", "ying", "yong", "you", "yu", "z", "zh", "xx", "sil"]
    
    dict_phone2id = {}
    dict_phone2id["PAD"] = 0
    for idx, phone in enumerate(all_phone):
        dict_phone2id[phone] = idx + 1  # avoid zero, since zero is used for padding.
    inv_map = {v: k for k, v in dict_phone2id.items()}

    phone_seq = []
    for onehot_feat in onehot_phone_seq:
        id_feat = np.nonzero(onehot_feat)[0][0]
        phone = inv_map[id_feat]
        phone_seq.append(phone)
        
    return phone_seq

def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data

def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning("Dataset in hdf5 file already exists. "
                                "recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error("Dataset in hdf5 file already exists. "
                              "if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


if __name__ == '__main__':
    phone_seq = ['sil', 'x', 'i', 'h', 'uan', 'y', 'ong', 'w', 'o', 'sil', 'd', 'e', 'y', 'in', 'd', 'iao']
    pho_dur = '2.0709999999999997 0.13 0.11111730532061792 0.08999999999999962 0.23288269467938205 0.065 0.17404253759854546 0.034999999999999996 0.2520702165906351 0.07712112695974559 0.06000000000000037 0.27876611885107383 0.08 0.19699999999999998 0.06 0.1781586512327075'
    fl = pl2fl(phone_seq, pho_dur, 22050, 256)
    pdb.set_trace()
