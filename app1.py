from flask import Flask
from flask import render_template
app = Flask(__name__)     # 建立Application物件

# 建立網站首頁的回應方式
@app.route("/")  # 創造出網域下名為"/"的網址
def index():    # view函式
    return "<h1>Hello Flask</h1>"   # 回傳網站首頁內容

app.run()