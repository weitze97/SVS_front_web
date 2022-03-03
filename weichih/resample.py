import soundfile as sf
from os.path import join, isfile
from os import listdir
import librosa
import pdb
import os

# pt_wav = './data/m1'
# re_sr = 22050
# list_song_number = [
#     f.split('.')[0] for f in listdir(pt_wav) 
#     if isfile(join(pt_wav, f)) 
#     and f.split('.')[-1] == 'wav'
#     ]
    
# for song_number in list_song_number:
#     print(song_number)
#     data, sr = librosa.load(join(pt_wav, song_number + '.wav'), sr=re_sr)
#     new_pt = join('temp', song_number + '.wav')
#     sf.write(new_pt, data, samplerate=re_sr)

pt_del = '/home/ahgmuse/svs/temp'
pt_mel = '/home/ahgmuse/svs/parallel_wavegan/dump_m1'
list_song_number = [
    f.split('.')[0] for f in listdir(pt_mel) 
    if isfile(join(pt_mel, f)) 
    and f.split('.')[-1] == 'h5'
    ]
list_del = [
    f.split('.')[0] for f in listdir(pt_del) 
    if isfile(join(pt_del, f)) 
    and f.split('.')[-1] == 'wav'
    ]
for song_number in list_song_number:
    if song_number in list_del:
        print(song_number)
        # pdb.set_trace()
        os.remove(join(pt_del, song_number+'.wav'))