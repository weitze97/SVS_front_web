from flask import Flask
from flask import render_template, request
from preprocess_v1_1_function import preprocess

app = Flask(__name__)     # 建立Application物件

# 建立網站首頁的回應方式
@app.route("/", methods=['GET', 'POST'])  # 創造出網域下名為"/"的網址
def home():
    if request.method == 'POST':
        if request.form.get('action1') == 'preprocess':
            preprocess()
            return render_template("home.html")
    elif request.method == 'GET':
        return render_template("home.html")   # 回傳網站首頁內容

#@app.route("/pred")
#def pred():
#    preprocess()
#    return render_template("pred.html")

if __name__ == "__main__":
    app.run() 