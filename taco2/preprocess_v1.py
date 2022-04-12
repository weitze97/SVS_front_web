import os
from os import listdir
from os.path import join, isfile
import numpy as np
import pandas as pd
import json
import pdb
from bopomofo.main import trans_sentense
from utils import load_filepaths_and_text
from utils_preprocess import *


tempo = 120
frame_unit = 0.005  # 5-ms base
sec_per_beat = 60/tempo
frame_per_beat = sec_per_beat/frame_unit
bopo2pinyin_path = 'filelists/dict_bopo2pinyin'
searching_pool_path = 'filelists/f1/searching_pool'
# sil 祝你生日快樂|1 58 58 60 58 63 62|2 0.5 0.5 1 1 1 2
input_path = 'filelists/f1/test.txt'  
output_path = input_path.split('.')[0] + '_pred.txt'
print(output_path)

with open(searching_pool_path, 'r') as f:
    searching_pool = json.load(f)

if os.path.exists(output_path):
    os.remove(output_path)

# w/ tonality consideration
dict_tone = {'-':'1', 'ˊ': '2', 'ˇ': '3', 'ˋ': '4', '˙': '5'}
dict_bopo2pinyin = import_bopo_pinyin(bopo2pinyin_path)
#pdb.set_trace()
# read input file
list_sent = load_filepaths_and_text(input_path)

count_rule0, count_rule1, count_rule2, count_rule3 = 0,0,0,0
for line in list_sent:
    lyrics, midis, notelens = line[0], line[1], line[2]

    # (1) language processor  
    lyrics = lyrics.split()
    list_char, list_bopo, list_dict = [], [], []
    for lyric in lyrics:
        if lyric == 'sil':
            list_char.append('sil')
            list_bopo.append('sil')
        else:
            list_char.extend(list(lyric)) 
            list_bopo.extend(trans_sentense(lyric).split(' '))

    # list_char = ['我', '歐', '弟', '歌', '聲', '穿', '安', '過', '深', '夜', 'sil']
    # list_bopo = ['ㄨㄛˇ', 'ㄡˇ', 'ㄊㄧˋ', 'ㄍㄜ-', 'ㄕㄥ-', 'ㄔㄨㄢ-', 'ㄢ-', 'ㄍㄨㄛˋ', 'ㄕㄣ-', 'ㄧㄝˋ', 'sil']

    # (2) Bopomofo to Pinyin
    sent_pho, sent_midi, sent_notelen, sent_pho_dur = '', '', '', ''
    for idx, bopo in enumerate(list_bopo):
        
        if 'sil' != bopo:
            # 輕聲:'˙ㄉㄜ' 會在最前面
            assert bopo[-1] in dict_tone.keys() or bopo[0] in dict_tone.keys()
            if bopo[-1] in dict_tone.keys():
                tone = dict_tone[bopo[-1]]
                bopo = bopo[0:-1] #remove the tone
            else:
                tone = dict_tone[bopo[0]]
                bopo = bopo[1:] #remove the tone
            
            pinyin_this = dict_bopo2pinyin[bopo]
            this_initial, this_final, _ = seperate_pinyin(pinyin_this)
        else:
            this_initial, this_final = bopo, ''

        sent_pho = sent_pho + this_initial + ' ' + this_final + ' '
    
    # (3) Duplicate to match the length of pho seq.
    assert len(list_char)==len(midis.split())==len(notelens.split()), (
        'Input Length Mismatch!')
    for idx, (midi, notelen) in enumerate(zip(midis.split(), notelens.split())):

        notelen = proper_round(float(notelen)*frame_per_beat) # beat unit to frame unit

        if list_char[idx] == 'sil':
            sent_midi = sent_midi + midi + ' '
            sent_notelen = sent_notelen + str(notelen) + ' ' 

            phoneme, midi, notelen, tone = ['sil'], [int(midi)], [notelen], ['xx']
        else:
            sent_midi = sent_midi + midi + ' ' + midi + ' '
            sent_notelen = sent_notelen + str(notelen) + ' ' + str(notelen) + ' '

            if list_bopo[idx][-1] in dict_tone.keys():
                tone = dict_tone[list_bopo[idx][-1]]
                bopo = list_bopo[idx][0:-1]  # remove the tone
            else:
                tone = dict_tone[list_bopo[idx][0]]
                bopo = list_bopo[idx][1:]  # remove the tone

            #tone = dict_tone[list_bopo[idx][-1]]
            #bopo = list_bopo[idx][0:-1]
            
            pinyin_this = dict_bopo2pinyin[bopo]
            this_initial, this_final, _ = seperate_pinyin(pinyin_this)
            phoneme = [this_initial, this_final]
            midi = [int(midi)]*len(phoneme)
            notelen = [notelen]*len(phoneme)
            tone = [tone]*len(phoneme)

        # (3) summarize in one sentence and a list of dict.
        list_dict.append(
            {'phoneme':phoneme, 
            'note':midi, 
            'note_len':notelen, 
            'tone':tone}
            )
    sentence = sent_pho + '|' + sent_midi + '|' + sent_notelen + '|'
    
    # (4) Duration estimation
    for char_item in list_dict:
        this_notelen = char_item['note_len'][0]

        if len(char_item['phoneme']) == 1:
            pho_dur_pred = char_item['note_len'][0]
            char_item['pho_dur_pred'] = [this_notelen*frame_unit]
            init_len, final_len = this_notelen*frame_unit, ''

        else:
            rule = rule_num(char_item['phoneme'], char_item['tone'], this_notelen, searching_pool)

            # (0) rule 0:
            if rule =='rule0':
                init_len, final_len = rule0(char_item['phoneme'], char_item['tone'], this_notelen, searching_pool)
                count_rule0 += 1

            # (1) rule 1:
            elif rule == 'rule1':
                init_len, final_len = rule1(char_item['phoneme'], this_notelen, searching_pool)
                count_rule1 += 1

            # (2) rule 2:
            elif rule == 'rule2':
                init_len, final_len = rule2(char_item['phoneme'], this_notelen, searching_pool)
                count_rule2 += 1

            # (3) rule 3:
            elif rule == 'rule3':
                init_len, final_len = rule3(char_item['phoneme'], this_notelen, searching_pool)
                count_rule3 += 1

            else:
                print("UNDER CONSTRUCT.")

            char_item['pho_dur_pred'] = [init_len, final_len]
            
        sent_pho_dur = sent_pho_dur + str(init_len) + ' ' + str(final_len) + ' '
    
    sentence = sentence + sent_pho_dur + '\n'
    with open(output_path, 'a') as f:
        f.write(sentence)
    
count_all_rule = count_rule0 + count_rule1 + count_rule2 + count_rule3
print('Estimation Finished!\nPercentage of terminating the search:')
print('rule0: {:.2f}'.format(count_rule0/(count_all_rule)*100))
print('rule1: {:.2f}'.format(count_rule1/(count_all_rule)*100))
print('rule2: {:.2f}'.format(count_rule2/(count_all_rule)*100))
print('rule3: {:.2f}'.format(count_rule3/(count_all_rule)*100))

'''
未處理:
(1) 若sil在中間
'''
    
