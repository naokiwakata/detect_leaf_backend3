from flask import Flask
from flask import Flask, render_template, request, make_response, jsonify
from flask import request, make_response, jsonify
import pandas as pd
import pickle
import util

from predictor import Predictor
from flask_cors import CORS

import cv2
from flask import Flask, render_template, request, make_response, jsonify
import numpy as np
import datetime
import shutil
import os

from predictor import Predictor
from flask_cors import CORS
import base64

import json

app = Flask(__name__)
CORS(app)

predictor = Predictor()


@app.route('/', methods=["GET", "POST"])
def predict_img(predictor: Predictor):
    # delete image
    shutil.rmtree('static/imgs/')
    os.mkdir('static/imgs/')

    if request.method == 'GET':
        img_path = None
        img_paths = None
    elif request.method == 'POST':

        file = request.files['file']

        stream = request.files['file'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        img = np.clip(img, 0, 255).astype(np.uint8)

        predictor.predict(img=img)
        img = predictor.img

        dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        _img_dir = "static/imgs/"
        img_path = _img_dir + dt_now + ".jpg"
        cv2.imwrite(img_path, img)

        predictor.processImage()
        img_paths = predictor.img_paths

    # 保存した画像ファイルのpathをHTMLに渡す
    return render_template('index.html', img_path=img_path, img_paths=img_paths)

@app.route("/trimming", methods=['GET', 'POST'])
def trimming(predictor : Predictor):
    shutil.rmtree('static/imgs/')
    os.mkdir('static/imgs/')
    data = request.get_json()
    post_img = data['post_img']
    img_base64 = post_img.split(',')[1]
    img_binary = base64.b64decode(img_base64)
    img_array = np.asarray(bytearray(img_binary), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    predictor.predict(img=img)
    predicted_img = predictor.img
    cv2.imwrite('result.jpg', predicted_img)

    with open('result.jpg', "rb") as f:
        predicted_img_base64 = base64.b64encode(f.read()).decode('utf-8')

    predictor.processImage()
    cut_img_paths = predictor.img_paths
    cut_img_base64_list = []

    for img_path in cut_img_paths:
        with open(img_path, "rb") as f:
            cut_img_base64 = base64.b64encode(f.read()).decode('utf-8')
            cut_img_base64_list.append(cut_img_base64)

    response = {'predict': predicted_img_base64,
                'cut_imgs': cut_img_base64_list, }
    return make_response(jsonify(response))



@app.route("/findHyperLeaf", methods=['GET', 'POST'])
def find():
    shutil.rmtree('static/imgs/')
    os.mkdir('static/imgs/')
    data = request.get_json()
    img_base64 = data['post_img']
    img_binary = base64.b64decode(img_base64)
    img_array = np.asarray(bytearray(img_binary), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    predictor.predict(img=img)
    predicted_img = predictor.img
    cv2.imwrite('result.jpg', predicted_img)

    pred_boxes = predictor.pred_boxes.tensor.to(
        'cpu').detach().numpy().tolist()
    pred_masks = predictor.pred_masks.to('cpu').detach().numpy().tolist()
    pred_classes = predictor.pred_classes.to('cpu').detach().numpy().tolist()

    response = {
                'Boxes': pred_boxes,
                'Masks': pred_masks,
                'Classes': pred_classes, 
                }
    print('dumpppp start')
    ## json.dumps の方が圧倒的に早い。余計なものがないからか？
    dump = json.dumps(response)
    print('dumpppp finish')

    ## めっちゃ遅い
    # jsonData = jsonify(response)


    print(len(pred_boxes))
    return make_response(dump)


@app.route("/judgeDisease", methods=['GET', 'POST'])
def judge():
    data = request.get_json()

    model_path = 'model\\svc_model.sav'
    model = pickle.load(open(model_path, 'rb'))

    predicts = {}
    for key, pixelSpectrals in data.items():
        data = pd.DataFrame(pixelSpectrals)
        print(key)
        # band11~51
        data = data.drop(columns=data.columns[53:63])  # バンドの端を削除
        data = data.drop(columns=data.columns[0:12])  # バンドの端を削除（いらない）
        data = data.reset_index(drop=True)  # indexを振りなおして元あるindexを削除
        data_average = data.describe().loc['mean']  # 平均値の取得
        standardized_data_frame = util.standardization(data_average)
        standardized_data_frame = pd.DataFrame(standardized_data_frame)

        predict = model.predict(standardized_data_frame.T)
        print(predict)
        predicts[key] = predict[0]
    response = {'Predicts': predicts}
    dump = json.dumps(response)
    return make_response(dump)


if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1', port=5000, threaded=True)
