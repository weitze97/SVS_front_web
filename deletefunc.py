import os
import glob
import shutil
#正式使用要將.txt改成.h5
def delete_mel() :
    melsdir = '/taco2/exp/pwg/mel/*.h5' #/home/undergrade_a/weichih/SVS_front_web/taco2/exp/pwg/mel/
    py_files = glob.glob(melsdir)
    for py_file in py_files:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
#正式使用要將.txt改成.wav
def delete_s() :
    soundfilesdir = 'static/exp/pwg/soundfile/*.wav'
    py_files = glob.glob(soundfilesdir)
    for py_file in py_files:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")