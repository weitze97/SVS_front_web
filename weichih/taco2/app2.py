from flask import Flask
from flask import render_template
app = Flask(__name__)     # 建立Application物件

# 建立網站首頁的回應方式
@app.route("/")  # 創造出網域下名為"/"的網址
def home():    # view函式
    return render_template("home.html")   # 回傳網站首頁內容

app.run() ##