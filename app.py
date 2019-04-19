from flask import Flask, jsonify,session
from flask import make_response
from flask import request
import base64,json
import cv2
import numpy as np
import base64
from text_analysis import get_pos_neg, render_word_cloud
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecret'

@app.route('/webservice/img', methods=['POST'])
def imagepro_upload_file():
    file = request.files['image']

    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    faces = detect_faces(image)
    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
        return jsonify('Görüntü işleme başarısız.')
    else:
        faceDetected = True
        num_faces = len(faces)

        for item in faces:
            draw_rectangle(image, item['rect'])

        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
        result={"Bulunan Yüz Sayısı":num_faces,"Resim":to_send}

        return jsonify(result)

def detect_faces(img):
    faces_list = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    if  len(faces) == 0:
        return faces_list

    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        face_dict = {}
        face_dict['face'] = gray[y:y + w, x:x + h]
        face_dict['rect'] = faces[i]
        faces_list.append(face_dict)
    return faces_list

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
######################################################################################################################################
########################################################Text Pro################################################################
######################################################################################################################################
@app.route('/webservice/text',methods = ['POST'])
def textpro():
        session['file_contents'] = request.form['text']
        file_contents = session.get('file_contents')
        analyzed_sent = get_pos_neg(file_contents)
        pos_sent = analyzed_sent[-1]
        neg_sent = analyzed_sent[0]
        word_cloud = render_word_cloud(file_contents)
        jsontext={"En Pozitif Cümle":pos_sent,"En Negatif Cümle":neg_sent,"Kullanılan Kelimeler(Resim Base 64)":word_cloud}
        post_control = 1
        return jsonify(jsontext)


if __name__ == '__main__':
    app.run()
