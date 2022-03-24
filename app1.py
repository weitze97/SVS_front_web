from flask import Flask
from flask import render_template, request, redirect, url_for, flash
from weichih.parallel_wavegan.bin.decode_function import decode
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)     # 建立Application物件

# 建立網站首頁的回應方式
@app.route("/", methods=['GET', 'POST'])  # 創造出網域下名為"/"的網址
def home():
    if request.method == 'POST':
        if request.form.get('decode') == 'decode':
            decode()
            return render_template("decode.html")
    return render_template("decode.html")

if __name__ == "__main__":
    app.run() 