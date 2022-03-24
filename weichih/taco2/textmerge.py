from os import close
import os
import re
'''
#def merge():
    #資料夾路徑記得更新到inputfiles存放的資料夾
    path01 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\01lyrics.txt'
    path02 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\02pitch.txt'
    path03 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\03notelength.txt'
    path04 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\04result.txt'
    path05 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\05text.txt'
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
        print('lyrics length : ', lyrics_total_line)
        print('pitch length : ', pitch_total_lines)
        print('notelength length : ', length_total_lines )
       
        with open ( path04, 'w' , encoding="utf-8") as f:
            print('input length do not match! cannot merge the text files!' , file=f )
            print('lyrics length : ', lyrics_total_line , file=f )
            print('pitch length : ', pitch_total_lines , file=f)
            print('notelength length : ', length_total_lines , file=f)

        
    else :
        with open(path05,'w',encoding="utf-8") as output:
            for i in range(max_line):
                if i==0 :
                    output.write(lyrics[i]+'|'+pitch[i]+'|'+length[i])
                else :
                    output.write('\n'+lyrics[i]+'|'+pitch[i]+'|'+length[i])
        
        with open ( path04, 'w' , encoding="utf-8") as f:
            print('textmerge seccess!' , file=f )
'''
#資料夾路徑記得更新到inputfiles存放的資料夾
#result 是合併成功和失敗原因顯示的紀錄
#text是合並完成後要進入preprocees前的檔案
path01 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\01lyrics.txt'
path02 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\02pitch.txt'
path03 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\03notelength.txt'
path04 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\04result.txt'
path05 = r'C:\nthu\SVMLAB\WEB\weitze\SVS_front_web\weichih\taco2\web_inputfiles\05text.txt'
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
    '''
        
    print('input length do not match!')
    print('lyrics length : ', lyrics_total_line)
    print('pitch length : ', pitch_total_lines)
    print('notelength length : ', length_total_lines )
       
    '''
    with open ( path04, 'w' , encoding="utf-8") as f:
        print('input length do not match! cannot merge the text files!' , file=f )
        print('lyrics length : ', lyrics_total_lines , file=f )
        print('pitch length : ', pitch_total_lines , file=f)
        print('notelength length : ', length_total_lines , file=f)
    with open ( path05, 'w' , encoding="utf-8") as f:
        print(' ' , file=f )


        
else :
    with open(path05,'w',encoding="utf-8") as output:
        for i in range(length_total_lines):
            if i==0 :
                output.write(lyrics[i]+'|'+pitch[i]+'|'+length[i])
            else :
                output.write('\n'+lyrics[i]+'|'+pitch[i]+'|'+length[i])
        
    with open ( path04, 'w' , encoding="utf-8") as f:
        print('textmerge seccess!' , file=f )
