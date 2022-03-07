from flask import Flask
from flask import render_template
app = Flask(__name__)     # 建立Application物件

@app.route('/data/appInfo/<name>', methods=['GET'])
def queryDataMessageByName(name):
    print("type(name) : ", type(name))
    return 'String => {}'.format(name)
app.run()