from os import close
import os
import re
#資料夾路徑記得更新到inputfiles存放的資料夾
path01 = r'C:\nthu\SVMLAB\WEB\web_aaply\textmerge\line_multiple\inputfiles\01lyrics.txt'
path02 = r'C:\nthu\SVMLAB\WEB\web_aaply\textmerge\line_multiple\inputfiles\02pitch.txt'
path03 = r'C:\nthu\SVMLAB\WEB\web_aaply\textmerge\line_multiple\inputfiles\03notelength.txt'

with open( path01 ,'r', encoding="utf-8") as f:
    lyrics = f.read().splitlines()
    lyrics_total_lines = len(lyrics)
    

with open( path02 ,'r', encoding="utf-8") as f:
    pitch = f.read().splitlines()
    pitch_total_lines = len(pitch)
    

with open( path03 ,'r', encoding="utf-8") as f:
    length = f.read().splitlines()
    length_total_lines = len(length)
    
if lyrics_total_lines != pitch_total_lines or pitch_total_lines != length_total_lines or lyrics_total_lines != length_total_lines :
    print('input length do not match!')
    print('lyrics length : ',lyrics_total_lines)
    print('pitch length : ',pitch_total_lines)
    print('notelemgth length : ',length_total_lines)
    
else :
    with open('text.txt','w',encoding="utf-8") as output:
        for i in range(max_line):
            if i==0 :
                output.write(lyrics[i]+'|'+pitch[i]+'|'+length[i])
            else :
                output.write('\n'+lyrics[i]+'|'+pitch[i]+'|'+length[i])
        


