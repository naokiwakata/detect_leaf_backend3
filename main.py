import cv2
from flask import Flask, render_template, request, make_response, jsonify
import numpy as np
import datetime
import shutil
import os

from predictor import Predictor
from fileEnum import File
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)

predictor = Predictor()


@app.route('/', methods=["GET", "POST"])
def predict_img():
    # delete image
    shutil.rmtree('static/imgs/')
    os.mkdir('static/imgs/')

    if request.method == 'GET':
        img_path = None
        img_paths = None
    elif request.method == 'POST':

        file = request.files['file']
        type = File.checkFileType(file=file)

        if(type == File.Image):
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
def parse():
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
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    response = {'result': img_base64}
    return make_response(jsonify(response))


if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1', port=5000)
