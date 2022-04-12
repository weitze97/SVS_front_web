import os
from os import listdir
from os.path import join, isfile
import numpy as np
import pandas as pd
import json


list_initial = ['b', 'p', 'm', 'f', 'd', 't', 'ng', 'n', 'l',
                'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh',
                'r', 'z', 'c', 's']
list_final = ['iamg', 'iang', 'iomg', 'iong', 'uamg', 'uang', 'uemg', 'ueng',
              'amg', 'ang', 'emg', 'eng', 'iam', 'ian', 'iao', 'img', 'ing', 'iou',
              'ong', 'uai', 'uam', 'uan', 'uei', 'uem', 'un', 'vam', 'van',
              'ai', 'am', 'an', 'ao', 'ei', 'em', 'en', 'ia', 'ie', 'im', 'in',
              'io', 'ou', 'ua', 'uo', 've', 'vm', 'vn', 'm', 'ng', 'n', 'a', 'o',
              'eh', 'e', 'ih', 'irh', 'i', 'u', 'v','er']  # +'er'

lasttable = {'ya': 'a', 'ia': 'a', 'ye': 'e', 'ie': 'e', 'yao': 'ao', 'iao': 'ao', 'you': 'ou','iou': 'ou', 
             'yan': 'an', 'ian': 'an', 'yin':'in', 'yang': 'ang', 'iang': 'ang','ying': 'ing',
             'wa': 'a', 'ua': 'a', 'wo':'o','uo': 'o', 'wai':'ai', 'uai': 'ai', 'wei': 'ei','uei': 'ei',
             'wan': 'an', 'uan': 'an', 'wen': 'en','uen': 'en', 'wang': 'ang', 'uang': 'ang', 'weng': 'eng', 'ong': 'eng',
             'yue': 'e', 'ue': 'e', 'yuan': 'an', 'uan': 'an', 'yun': 'en', 'un': 'en', 'yong': 'eng', 'iong': 'eng',
             'irh': 'i', 'ih': 'i'
            }

def seperate_pinyin(pinyin_word):
    
    two_initial = pinyin_word[0:2]
    
    if pinyin_word in list_final:
        final = pinyin_word
        initial = '$0'  # zero-initial token
    
    elif two_initial in list_initial:
        initial = two_initial
        final = pinyin_word[2:]
        
    else:
        initial = pinyin_word[0]
        final = pinyin_word[1:]
        
    if final not in list_final:
        print(pinyin_word, final)
        raise 'Error!'
    else:
        if final in lasttable.keys():
            last = lasttable[final]
        else:
            last = final
        
    return initial, final, last

def import_bopo_pinyin(path_csv):
    with open(path_csv, 'r') as f:
        bopo_pinyin = json.load(f)
        
    dict_bopo2pinyin = {}
    
    for bopo, pinyin in bopo_pinyin.items():
        if bopo != '':
            dict_bopo2pinyin[bopo] = pinyin
    
    return dict_bopo2pinyin

def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return int(float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1)))
    return int(float(num[:-1]))


# --------------------- rule-base Algorithm ----------------------

def rule0(phonemes, tone, this_notelen, list_all_text, frame_len=0.005):
    list_sameP_notelen = [(e['note_len'][0], e['pho_dur']) for i, e in enumerate(list_all_text) 
                          if e['phoneme'] == phonemes and e['tone'] == tone]
    same_notelen = []
    sim_notelen = []
    for nl, phs in list_sameP_notelen:
        i_ratio = phs[0] / (this_notelen*frame_len)
        f_ratio = phs[1] / (this_notelen*frame_len)
            
        # (1) 先找一模一樣的
        if abs(nl - this_notelen) == 0:
            same_notelen.append((nl, [i_ratio, f_ratio]))

        # (2) 此規則最多容忍到5個frames, 0.025秒
        elif abs(nl - this_notelen) <= 5:
            sim_notelen.append((nl, [i_ratio, f_ratio]))
        else:
            continue
            
    if same_notelen != []:
        mean_i_ratio = np.mean([e[1][0] for e in same_notelen])
        init_len = mean_i_ratio * (this_notelen*frame_len)
        final_len = (1-mean_i_ratio) * (this_notelen*frame_len)
    else:
        mean_i_ratio = np.mean([e[1][0] for e in sim_notelen])
        init_len = mean_i_ratio * (this_notelen*frame_len)
        final_len = (1-mean_i_ratio) * (this_notelen*frame_len)
    
    return init_len, final_len


def rule1(phonemes, this_notelen, list_all_text, frame_len=0.005):
    list_sameP_notelen = [(e['note_len'][0], e['pho_dur']) for i, e in enumerate(list_all_text) if e['phoneme'] == phonemes]
    same_notelen = []
    sim_notelen = []
    for nl, phs in list_sameP_notelen:
        i_ratio = phs[0] / (this_notelen*frame_len)
        f_ratio = phs[1] / (this_notelen*frame_len)
            
        # (1) 先找一模一樣的
        if abs(nl - this_notelen) == 0:
            same_notelen.append((nl, [i_ratio, f_ratio]))

        # (2) 此規則最多容忍到5個frames, 0.025秒
        elif abs(nl - this_notelen) <= 5:
            sim_notelen.append((nl, [i_ratio, f_ratio]))
        else:
            continue
            
    if same_notelen != []:
        mean_i_ratio = np.mean([e[1][0] for e in same_notelen])
        init_len = mean_i_ratio * (this_notelen*frame_len)
        final_len = (1-mean_i_ratio) * (this_notelen*frame_len)
    else:
        mean_i_ratio = np.mean([e[1][0] for e in sim_notelen])
        init_len = mean_i_ratio * (this_notelen*frame_len)
        final_len = (1-mean_i_ratio) * (this_notelen*frame_len)
    
    return init_len, final_len


def rule2(phonemes, this_notelen, list_all_text, frame_len=0.005):
    list_sameIni_notelen = [(e['note_len'][0], e['pho_dur']) for i, e in enumerate(list_all_text) if e['phoneme'][0] == phonemes[0]]
    sim_notelen = []
    for nl, phs in list_sameIni_notelen:
        i_ratio = phs[0] / (this_notelen*frame_len)
        f_ratio = phs[1] / (this_notelen*frame_len)

        # 此規則最多容忍到5個frames, 0.025秒
        if abs(nl - this_notelen) <= 5:
            sim_notelen.append((nl, [i_ratio, f_ratio]))
        else:
            continue
            
    mean_i_ratio = np.mean([e[1][0] for e in sim_notelen])
    init_len = mean_i_ratio * (this_notelen*frame_len)
    final_len = (1-mean_i_ratio) * (this_notelen*frame_len)
    
    return init_len, final_len


def rule3(phonemes, this_notelen, list_all_text, frame_len=0.005):
    # 拉得太長，從母音下手
    list_sameIni_notelen = [(e['note_len'][0], e['pho_dur']) for i, e in enumerate(list_all_text) if e['phoneme'][0] == phonemes[0]]

    num_frames = [i*10 for i in range(1, 11)]
    dict_std_mean = {k: {'sim_notelen':[], 'std':None, 'mean':None} for k in num_frames}
    for num_frame in num_frames:
          
        for nl, phs in list_sameIni_notelen:
            i_ratio = phs[0] / (this_notelen*frame_len)
            f_ratio = phs[1] / (this_notelen*frame_len)

            # 10s-10s往上加，找std最小的
            if abs(nl - this_notelen) <= num_frame:
                dict_std_mean[num_frame]['sim_notelen'].append((nl, [i_ratio, f_ratio]))
            else:
                continue
        
        dict_std_mean[num_frame]['mean'] = np.mean([e[1][0] for e in dict_std_mean[num_frame]['sim_notelen']])
        dict_std_mean[num_frame]['std'] = np.std([e[1][0] for e in dict_std_mean[num_frame]['sim_notelen']])

    min_std_frame = (np.nanargmin([dict_std_mean[n]['std'] for n in num_frames])+1)*10
    
    mean_i_ratio = dict_std_mean[min_std_frame]['mean']
    init_len = mean_i_ratio * (this_notelen*frame_len)
    final_len = (1-mean_i_ratio) * (this_notelen*frame_len)

    return init_len, final_len


def rule_num(phonemes, tone, this_notelen, list_all_text):
    
    # (0) is rule0?
    list_sameP_tone_notelen = [(e['note_len'][0], e['pho_dur']) for i, e in enumerate(list_all_text) 
                                  if e['phoneme'] == phonemes and e['tone'] == tone]
    same_notelen = []
    sim_notelen = []
    for nl, phs in list_sameP_tone_notelen:
        i_ratio = phs[0] / (this_notelen*0.005)
        f_ratio = phs[1] / (this_notelen*0.005)
            
        # (0-1) 先找一模一樣的
        if abs(nl - this_notelen) == 0:
            same_notelen.append((nl, [i_ratio, f_ratio]))

        # (0-2) 此規則最多容忍到5個frames, 0.025秒
        elif abs(nl - this_notelen) <= 5:
            sim_notelen.append((nl, [i_ratio, f_ratio]))
        else:
            continue
            
    if same_notelen != [] or sim_notelen != []:
        return 'rule0'
    
    else:
        # (1) is rule1 ?
        list_sameP_notelen = [(e['note_len'][0], e['pho_dur']) for i, e in enumerate(list_all_text) if e['phoneme'] == phonemes]
        same_notelen = []
        sim_notelen = []
        for nl, phs in list_sameP_notelen:
            i_ratio = phs[0] / (this_notelen*0.005)
            f_ratio = phs[1] / (this_notelen*0.005)

            # (1-1) 先找一模一樣的
            if abs(nl - this_notelen) == 0:
                same_notelen.append((nl, [i_ratio, f_ratio]))

            # (1-2) 此規則最多容忍到5個frames, 0.025秒
            elif abs(nl - this_notelen) <= 5:
                sim_notelen.append((nl, [i_ratio, f_ratio]))
            else:
                continue

        if same_notelen != [] or sim_notelen != []:
            return 'rule1'

        else:
            # (2) is rule2 ?
            list_sameIni_notelen = [(e['note_len'][0], e['pho_dur']) for i, e in enumerate(list_all_text) if e['phoneme'][0] == phonemes[0]]
            sim_notelen = []
            for nl, phs in list_sameIni_notelen:
                i_ratio = phs[0] / (this_notelen*0.005)
                f_ratio = phs[1] / (this_notelen*0.005)

                # (2-1) 此規則最多容忍到5個frames, 0.025秒
                if abs(nl - this_notelen) <= 5:
                    sim_notelen.append((nl, [i_ratio, f_ratio]))
                else:
                    continue

            if sim_notelen != []:
                return 'rule2'

            else:
                return 'rule3'