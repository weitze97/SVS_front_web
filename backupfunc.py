import os
import glob
import shutil

#假設合成音檔finalfiles名稱為此處的test1.txt
#從網站獲取new_name input
#正式使用要將.txt改成.wav
def backup_s(new_name) :
    src = 'static/exp/pwg/soundfile/pwg_birthday.wav'
    #des_0 = 'static/exp/pwg/backup/soundfiles_b'
    des_1 = 'static/exp/pwg/soundfiles_b/'+new_name+'.wav'
    #dir_files = glob.glob(des_0 + '/*.txt')
    os.rename(src,des_1)

#os.rename的錯誤會讓整個py檔停止執行
#成功返回None字串

