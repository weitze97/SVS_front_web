from flask import Flask
from flask import render_template, request, redirect, url_for, flash, abort
from taco2.preprocess_v1_1_function import preprocess
from taco2.synth_function import synth
from parallel_wavegan.bin.decode_function import decode
from web_inputfiles.textmerge import merge
import os
from werkzeug.utils import secure_filename
from backupfunc import backup_s
import deletefunc as df

app = Flask(__name__)     # 建立Application物件

UPLOAD_FOLDER_1 = './taco2/filelists/f1'

ALLOWED_EXTENSIONS = {'txt', 'bak'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_1


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# 建立網站首頁的回應方式
@app.route("/", methods=['GET', 'POST'])  # 創造出網域下名為"/"的網址
def home():
    if request.method == 'GET':
        return render_template("home_v3.html")   # 回傳網站首頁內容
    return render_template("home_v3.html")

@app.route("/oneclick", methods=['GET', 'POST'])
def oneclick():
    if request.method == 'POST':
        if request.form.get('upload') == '上傳單一檔案':
            file = request.files['f_done']
            if file and allowed_file(file.filename):
                # 上傳檔案到目標資料夾
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'test.txt'))
                return render_template("oneclick.html")
        if request.form.get('start') == '一鍵合成！':
            return redirect(url_for('svs_process'))
    return render_template("oneclick.html")


@app.route("/svs_process", methods=['GET', 'POST'])
def svs_process():
    if request.method == 'GET':
        #abort(500) #手動製造錯誤
        df.delete_mel()
        df.delete_s()
        preprocess()
        synth()
        decode()
        return redirect(url_for('home'))
    return render_template("svs_process.html")

@app.route("/mergefile", methods=['GET', 'POST'])
def mergefile():
    UPLOAD_FOLDER_2 = './web_inputfiles'
    app.config['UPLOAD_FOLDER_MERGE'] = UPLOAD_FOLDER_2
    if request.method == 'POST':
        if request.form.get('upload') == '上傳歌詞':
            file = request.files['f_lyrics']
            if  file and allowed_file(file.filename):
                # 上傳檔案到目標資料夾
                file.save(os.path.join(app.config['UPLOAD_FOLDER_MERGE'], '01lyrics.txt'))
                return render_template("mergefile.html")

        if request.form.get('upload') == '上傳音高':
            file = request.files['f_pitch']
            if  file and allowed_file(file.filename):
                # 上傳檔案到目標資料夾
                file.save(os.path.join(app.config['UPLOAD_FOLDER_MERGE'], '02pitch.txt'))
                return render_template("mergefile.html")

        if request.form.get('upload') == '上傳音長':
            file = request.files['f_notelen']
            if  file and allowed_file(file.filename):
                # 上傳檔案到目標資料夾
                file.save(os.path.join(app.config['UPLOAD_FOLDER_MERGE'], '03notelength.txt'))
                return render_template("mergefile.html")

        if request.form.get('start') == '一鍵合併並合成！':
            merge()
            return redirect(url_for('svs_process'))
    return render_template("mergefile.html")


@app.route("/music/", methods=['GET', 'POST'])
def music():
    his_songs = os.listdir('static/exp/pwg/soundfiles_b/')
    songs = os.listdir('static/exp/pwg/soundfile/')
    if request.method == 'GET':
        return render_template("music.html", songs=songs , his_songs=his_songs)
    elif request.method == 'POST':
        if request.form.get('save') == 'save':
            #獲取使用者取名
            new_name = request.form["nm"]
            backup_s(new_name)
            his_songs = os.listdir('static/exp/pwg/soundfiles_b/')
            songs = os.listdir('static/exp/pwg/soundfile/')
            return render_template("music.html", songs=songs , his_songs=his_songs)
        #討論後決定要不要留，amy覺得可以拿掉
        if request.form.get('clear') == 'clear':
            #df.delete_mel()
            df.delete_s()
            his_songs = os.listdir('static/exp/pwg/soundfiles_b/')
            songs = os.listdir('static/exp/pwg/soundfile/')
            return render_template("music.html", songs=songs , his_songs=his_songs)
    return render_template("music.html", songs=songs , his_songs=his_songs)
    
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 檢查POST有沒有符合檔名
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # 如果沒有選檔案，瀏覽器會送出一個沒有檔名的檔案
        if  file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if  file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # 上傳檔案到目標資料夾
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file'))
            #return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html> 
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

# Error Handlers
@app.errorhandler(404)
def not_found(e):
    return render_template("errorpage/404.html")

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {e}, route: {request.url}")
    return render_template("errorpage/500.html")


if __name__ == "__main__":
    app.run(debug=False) 
